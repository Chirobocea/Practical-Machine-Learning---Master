#Source code: https://huggingface.co/blog/how-to-train

# # We won't need TensorFlow here
# !pip uninstall -y tensorflow
# # Install `transformers` from master
# !pip install git+https://github.com/huggingface/transformers
# !pip list | grep -E 'transformers|tokenizers'
# # transformers version at notebook update --- 2.11.0
# # tokenizers version at notebook update --- 0.8.0rc1

%%time
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# corpus.txt contains all the text from all available jsons (train, valdiation, test)
paths = ["/kaggle/input/corpus/corpus.txt"]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=15_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

!mkdir MyTokenizer15k
# save the tokenizer
tokenizer.save_model("/kaggle/working/MyTokenizer15k")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "/kaggle/input/mytokenizer15k/vocab.json",
    "/kaggle/input/mytokenizer15k/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

from transformers import RobertaConfig

# define own roberta-like architecture
config = RobertaConfig(
    vocab_size=15_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

# load trained tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("/kaggle/input/mytokenizer15k", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()

%%time
from transformers import LineByLineTextDataset

# define dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/kaggle/input/corpus/train.txt",
    block_size=512,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# !pip uninstall accelerate
# !pip uninstall transformers
# !pip install accelerate>=0.20.1
# !pip install transformers[torch]>=4.10.0

from transformers import Trainer, TrainingArguments
import wandb
import os

# Set your WandB API key
#os.environ["WANDB_API_KEY"] =

# define training arguments
training_args = TrainingArguments(
    output_dir="MyModel15k",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
)

# use hugging faces trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

%%time
trainer.train()

trainer.save_model("MyModel15k")