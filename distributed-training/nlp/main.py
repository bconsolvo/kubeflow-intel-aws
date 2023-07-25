from trainer import Trainer
from omegaconf import DictConfig
from model import GPT
import hydra
import time


def get_time(total_time):
    # Convert total time to hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Format time as hh:mm:ss
    formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return formatted_time


def get_model_and_opt(model_config, opt_config):
    # determine the vocab size we'll use for from-scratch training
    if model_config.vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
        model_config.vocab_size = 50304
    print(model_config)

    print(f"Initializing from OpenAI GPT-2 weights: gpt2")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=model_config.dropout)
    model = GPT.from_pretrained("gpt2", override_args)

    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_config[k] = getattr(model.config, k)

    # crop down the model block size if desired, using model surgery
    # so that the checkpoint will have the right value
    if model_config.block_size < model.config.block_size:
        model.crop_block_size(model_config.block_size)
        model_config["block_size"] = model_config.block_size

    optimizer = model.configure_optimizers(
        opt_config.weight_decay,
        opt_config.learning_rate,
        (opt_config.beta1, opt_config.beta2),
        "cpu",
    )
    return model, optimizer


@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    start = time.time()

    model, optimizer = get_model_and_opt(cfg.model_config, cfg.optimizer_config)
    trainer = Trainer(
        cfg.trainer_config, cfg.data_dir, model, optimizer, cfg.model_config.block_size
    )
    trainer.train()
    end = time.time()

    print(f"Training completed! Total time taken: {get_time(end-start)}")


if __name__ == "__main__":
    main()
