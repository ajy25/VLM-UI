from vlmui.models.xraygpt.download import download_xraygpt
from vlmui.models.CheXagent.download import download_CheXagent


def download_models(model_names: list[str]):
    for model_name in model_names:
        if model_name == "xraygpt":
            download_xraygpt()
        elif model_name == "CheXagent":
            download_CheXagent()
        else:
            raise ValueError(f"Model {model_name} not found.")


if __name__ == "__main__":
    download_models(["CheXagent"])
