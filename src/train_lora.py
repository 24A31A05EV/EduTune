import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "distilgpt2"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "models/edutune-lora"


def main():

    print("=" * 60)
    print("EDUTUNE - LoRA TRAINING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.config.pad_token_id = tokenizer.pad_token_id

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    print("\nConfiguring LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],
    )

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=1,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=1,

        learning_rate=2e-4,

        logging_steps=1,

        save_strategy="epoch",

        eval_strategy="epoch",

        report_to="none",

        use_cpu=True,

        max_length=256,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING LORA TRAINING")
    print("=" * 60)

    trainer.train()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print("\nSaving EduTune LoRA adapter...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()