from huggingface_hub import HfApi

repo_id = "Nikod3m/hacknation-2025-NASK-herbert-ner-v2"
folder_path = "./models/herbert_ner_v2"

api = HfApi()

api.create_repo(repo_id=repo_id, exist_ok=True)

print(f"Wysyłanie plików z {folder_path} do {repo_id}...")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    path_in_repo=".",
    ignore_patterns=["checkpoint-*", "training_args.bin"],
    commit_message="Upload fine-tuned HerBERT NER model files"
)

print("Gotowe! Model jest na Hugging Face.")