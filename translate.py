import tqdm, json
from datasets import load_dataset
from os import environ
from google.cloud import translate
from multiprocessing import Pool

PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

# Load the MRPC dataset from Hugging Face
dataset = load_dataset("glue", "mrpc")
train_data, validation_data, test_data = [dataset[split].to_list() for split in dataset.keys()]


def translate_sentence_googlecloud(
    text: str, source_language_code="en", target_language_code="az"
) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )
    return response.translations[0].translated_text


# translate one example
def translate_example(example):
    return {
        "sentence1": translate_sentence_googlecloud(example["sentence1"]),
        "sentence2": translate_sentence_googlecloud(example["sentence2"]),
        "label": example["label"],
        "idx": example["idx"],
    }


# translate examples sequentially one at a time.
def translate_split(split):
    translated_split = []
    for example in tqdm.tqdm(split):
        translated_example = translate_example(example)
        translated_split.append(translated_example)
    return translated_split


# translated parallely accross `num_processes` cpus
def translate_split_parallel(split, num_processes=6):
    with Pool(num_processes) as pool:
        translated_split = list(tqdm.tqdm(pool.imap(translate_example, split), total=len(split)))
    return translated_split


# dump the dataset into json file
def write_data(split, data):
    with open(f"{split}.jsonl", "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def main():
    translated_train = translate_split_parallel(train_data)
    write_data("train", translated_train)
    translated_validation = translate_split_parallel(validation_data)
    write_data("validation", translated_validation)
    translated_test = translate_split_parallel(test_data)
    write_data("test", translated_test)


if __name__ == "__main__":
    # main()
    translate_sentence_googlecloud("hello man whats up?")

    # data_lengths = lambda data: sum(len(item["sentence1"] + item["sentence2"]) for item in data)
    # total_length = data_lengths(train_data) + data_lengths(validation_data) + data_lengths(test_data)
    # print(total_length)

    # text = "Hello man, what are you doing now?"
    # translated = translate_sentence_googlecloud(text)
    # print(translated)
