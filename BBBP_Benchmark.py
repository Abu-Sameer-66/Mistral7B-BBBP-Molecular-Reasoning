# Installation of required libraries for LLM Fine-tuning
!pip install --quiet deepchem rdkit transformers peft bitsandbytes accelerate

import os, gc, torch, pandas as pd, numpy as np, deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# System and memory management for stable training on T4 GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

# --- 1. DATASET PREPARATION & SANITIZATION ---
# Loading the BBBP dataset from DeepChem's S3 bucket
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df_raw = pd.read_csv(url)

# Sanitization: Filtering out 11 molecules with invalid Nitrogen valences (valence > 4) 
# to ensure chemical validity and prevent numerical instability.
df_clean = df_raw[df_raw['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)].copy()

# Implementing Scaffold Split for rigorous evaluation on out-of-distribution structures
dataset = dc.data.NumpyDataset(X=np.zeros(len(df_clean)), ids=df_clean['smiles'].values)
train_idx, _, test_idx = dc.splits.ScaffoldSplitter().split(dataset)

def build_property_aware_dataset(df, indices, is_train=True):
    subset = df.iloc[indices]
    data = []
    for _, row in subset.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        # Calculating TPSA and Molecular Weight as key drivers for BBB permeability
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Scientific Prompting: Providing physical context to guide the model's latent reasoning
        prompt = (f"Analysis: Brain Barrier Penetration\n"
                  f"Descriptors: MW={mw:.1f}, TPSA={tpsa:.1f}\n"
                  f"SMILES: {row['smiles']}\n"
                  f"Permeable: {'Yes' if int(row['p_np']) == 1 else 'No'}")
        
        # Addressing class imbalance: Oversampling the minority 'No' class (3x to 4x)
        repeat = 4 if (is_train and int(row['p_np']) == 0) else 1
        for _ in range(repeat):
            data.append({'prompt': prompt, 'label': int(row['p_np'])})
    return pd.DataFrame(data)

df_train = build_property_aware_dataset(df_clean, train_idx).sample(frac=1)
df_test = build_property_aware_dataset(df_clean, test_idx, is_train=False)

# --- 2. MODEL ARCHITECTURE (Mistral-7B + QLoRA) ---
MODEL_ID = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit NF4 Quantization to optimize memory for 16GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
base_model.gradient_checkpointing_enable()

# Using LoRA Rank 32 to capture complex chemical relationships in the BBBP task
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(base_model, peft_config)
model.enable_input_require_grads()

# --- 3. TRAINING & VALIDATION LOOP ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
id_yes = tokenizer.encode("Yes", add_special_tokens=False)[-1]
id_no = tokenizer.encode("No", add_special_tokens=False)[-1]

total_steps = 2500
current_step = 0
best_auc = 0
train_ds = dc.data.NumpyDataset(X=df_train['prompt'].values)

print("[*] Starting step-based fine-tuning...")
while current_step < total_steps:
    for (X_batch, _, _, _) in train_ds.iterbatches(batch_size=2):
        model.train()
        tokens = tokenizer(X_batch.tolist(), padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
        
        loss = model(**tokens, labels=tokens["input_ids"]).loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Periodic validation every 250 steps to monitor generalization on scaffold split
        if current_step % 250 == 0 and current_step > 0:
            model.eval()
            y_probs, y_true = [], []
            with torch.no_grad():
                for _, row in df_test.iterrows():
                    eval_p = row['prompt'].split("Permeable:")[0] + "Permeable:"
                    inputs = tokenizer(eval_p, return_tensors='pt').to("cuda")
                    logits = model(input_ids=inputs['input_ids']).logits[0, -1, [id_no, id_yes]]
                    y_probs.append(F.softmax(logits, dim=-1)[1].item())
                    y_true.append(row['label'])
            
            auc = roc_auc_score(y_true, y_probs)
            print(f"Step {current_step} | Loss: {loss.item():.4f} | Validation ROC-AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                model.save_pretrained("./best_bbbp_model")
        
        current_step += 1
        if current_step >= total_steps: break
