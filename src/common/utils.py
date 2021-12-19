import shutil
import importlib
import torch
import torch.nn.functional as F

def get_model_class(cmd, copy=True):
    """
    Get the model class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Model class
    """
    model_module_str = ".".join(cmd.model.split(".")[:-1])
    model_class_str = cmd.model.split(".")[-1]

    model_module = importlib.import_module(model_module_str)
    if copy:
        module_fpath = model_module.__file__
        shutil.copy(module_fpath, cmd.experiment_dpath)
    model_class = getattr(model_module, model_class_str)

    return model_class


def get_dataset_class(cmd, copy=True):
    """
    Get the Dataset class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Dataset class
    """
    dataset_module_str = ".".join(cmd.dataset.split(".")[:-1])
    dataset_class_str = cmd.dataset.split(".")[-1]

    dataset_module = importlib.import_module(dataset_module_str)

    if copy:
        dataset_module_fpath = dataset_module.__file__
        shutil.copy(dataset_module_fpath, cmd.experiment_dpath)
    dataset_class = getattr(dataset_module, dataset_class_str)

    return dataset_class

def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def generate_images(text, model, filter_thres, temperature=1.):
    vae, text_seq_len, image_seq_len, num_text_tokens = model.vae, \
                                                        model.text_seq_len, \
                                                        model.image_seq_len, \
                                                        model.num_text_tokens
    total_len = text_seq_len + image_seq_len

    text = text[:, :text_seq_len]  # make sure text is within bounds
    out = text

    for cur_len in range(out.shape[1], total_len):
        is_image = cur_len >= text_seq_len
        text, image = out[:, :text_seq_len], out[:, text_seq_len:]
        logits = model(text, image)[:, -1, :]
        filtered_logits = top_k(logits, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)

        sample -= (num_text_tokens if is_image else 0)
        out = torch.cat((out, sample), dim=-1)

    img_seq = out[:, -image_seq_len:]
    images = vae.decode(img_seq)

    return images, img_seq