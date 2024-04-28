import sys
sys.path.append("../")
from utils import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments
from idefics2.idefics_ds import main

if __name__ == "main":
  parser = HfArgumentParser(
      (ModelArguments, DataTrainingArguments, TrainingArguments)
  )
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
      # If we pass only one argument to the script and it's the path to a json file,
      # let's parse it to get our arguments.
      model_args, data_args, training_args = parser.parse_json_file(
          json_file=os.path.abspath(sys.argv[1])
      )
  else:
      model_args, data_args, training_args = parser.parse_args_into_dataclasses()
  main(model_args, data_args, training_args)