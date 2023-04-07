import shutil

import torch
from tokenizers import ByteLevelBPETokenizer

from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TextDataset,
    Trainer,
    TrainingArguments,
)


def main():
    vocab_size = 32768

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=["../../datasets/facebook_messages/messages.txt"],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
        ],
    )
    tokenizer.save_model("tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")

    config = GPT2Config.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048,
    )
    model = GPT2LMHeadModel(config)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="../../datasets/facebook_messages/messages.txt",
        block_size=256,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    shutil.rmtree("./results", ignore_errors=True)

    training_args = TrainingArguments(
        output_dir="./results",
        save_steps=100,
        num_train_epochs=80,
        per_device_train_batch_size=32,
        logging_steps=10,
        save_total_limit=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("saved_model/mond/slovak-fb-msg-gpt2")


if __name__ == "__main__":
    main()
