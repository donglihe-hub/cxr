from huggingface_hub.utils import HfFolder

if HfFolder.get_token() is None:
    from huggingface_hub import login
    login()
else:
    print("Token already set")
