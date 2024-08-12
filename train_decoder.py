import torch
import torch.nn as nn
from os.path import isfile
from train_clip import train_clip
from train_prior import train_prior
from torch.utils.data import DataLoader
from data.FMNISTConfig import FMNISTConfig
from torch.optim import Adam, AdamW, lr_scheduler
from data.dataset import get_train_set, get_test_set
from model.decoder import Decoder, sample_plot_image
from data.data_utils import get_schedule_values, forward_diffusion, tokenizer

def train_decoder(config):
    train_set, mean, std = get_train_set(config, augment_data=config.decoder.augment_data)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.decoder.batch_size, num_workers=config.decoder.num_workers)

    if config.decoder.validate:
        val_set = get_test_set(config, mean=mean, std=std)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=config.decoder.batch_size, num_workers=config.decoder.num_workers)

    schedule_values = get_schedule_values(schedule=config.decoder.schedule, max_time=config.decoder.max_time, device=config.device)

    decoder = Decoder(config).to(config.device)

    if config.decoder.weight_decay == 0:
        optimizer = Adam(decoder.parameters(), lr=config.decoder.lr)
    else:
        optimizer = AdamW(decoder.parameters(), lr=config.decoder.lr, weight_decay=config.decoder.weight_decay)

    if config.decoder.warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.decoder.warmup_epochs), end_factor=1.0, total_iters=(config.decoder.warmup_epochs - 1), last_epoch=-1)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.decoder.epochs - config.decoder.warmup_epochs), eta_min=config.decoder.lr_min)

    if config.decoder.sample_after_epoch:
        sample_captions = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[0] for x in train_set.captions.values()]).to(config.device)
        sample_masks = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[1] for x in train_set.captions.values()]).to(config.device)

    best_loss = float('inf')
    for epoch in range(config.decoder.epochs):
        # Training
        decoder.train()
        training_loss = 0.0
        for batch in train_loader:
            image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)
            optimizer.zero_grad()

            # Calculating Loss
            timesteps = torch.randint(0, config.decoder.max_time, (image.shape[0],), device=config.device, dtype=torch.long)
            noisy_image, noise = forward_diffusion(image, schedule_values, timesteps)
            pred_noise = decoder(noisy_image, timesteps, caption, mask)
            loss = nn.functional.mse_loss(pred_noise, noise)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=config.decoder.grad_max_norm)
            optimizer.step()
            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)

        if epoch < config.decoder.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        # Validation
        if config.decoder.validate:
            decoder.eval()
            validation_loss = 0.0
            for batch in val_loader:
                image, caption, mask = batch["image"].to(config.device), batch["caption"].to(config.device), batch["mask"].to(config.device)

                # Calculating Loss
                timesteps = torch.randint(0, config.decoder.max_time, (image.shape[0],), device=config.device, dtype=torch.long)
                noisy_image, noise = forward_diffusion(image, schedule_values, timesteps)
                pred_noise = decoder(noisy_image, timesteps, caption, mask)
                loss = nn.functional.mse_loss(pred_noise, noise)
                validation_loss += loss.item()

            validation_loss = validation_loss / len(val_loader)

            if validation_loss <= best_loss:
                best_loss = validation_loss
                torch.save(decoder.state_dict(), config.decoder.model_location)

            print(f"[Epoch {epoch + 1}/{config.decoder.epochs}] Training Loss: {training_loss:.5f} | Validation Loss: {validation_loss:.5f}")
        else:
            torch.save(decoder.state_dict(), config.decoder.model_location)
            print(f"[Epoch {epoch + 1}/{config.decoder.epochs}] Training Loss: {training_loss:.5f}")

        if config.decoder.sample_after_epoch:
            caption = sample_captions[None, (epoch % len(sample_captions))]
            mask = sample_masks[None, (epoch % len(sample_masks))]
            sample_plot_image(config, caption, mask, schedule_values=schedule_values, decoder=decoder)

if __name__=="__main__":
    config = FMNISTConfig()

    if not isfile(config.clip.model_location):
        print("CLIP model has not been trained. Training CLIP...")
        print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if torch.cuda.is_available() else "")
        train_clip(config)

    if not isfile(config.prior.model_location):
        print("Prior model has not been trained. Training Prior...")
        print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if torch.cuda.is_available() else "")
        train_prior(config)

    print("Training Decoder...")
    print("Using device: ", config.device, f"({torch.cuda.get_device_name(config.device)})" if torch.cuda.is_available() else "")
    train_decoder(config)