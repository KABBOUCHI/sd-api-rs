use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use image::{ImageBuffer, Rgb};
use std::{collections::HashMap, vec};
use tch::{nn::Module, Device, Kind, Tensor};

const GUIDANCE_SCALE: f64 = 7.5;

#[get("/")]
async fn hello(params: web::Query<HashMap<String, String>>) -> impl Responder {
    let prompt = params.get("prompt");
    let prompt = prompt.unwrap();

    println!("MPS available: {}", tch::utils::has_mps());

    tch::maybe_init_cuda();

    let seed = 32;
    let n_steps = 30;
    let num_samples = 1;
    let vocab_file = "data/bpe_simple_vocab_16e6.txt".to_string();
    let clip_weights = "data/pytorch_model.safetensors".to_string();
    let unet_weights = "data/unet.safetensors".to_string();
    let vae_weights = "data/vae.safetensors".to_string();

    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, None, None);

    let device_setup = diffusers::utils::DeviceSetup::new(vec![]);
    let clip_device = device_setup.get("clip");
    let vae_device = device_setup.get("vae");
    let unet_device = device_setup.get("unet");
    let scheduler = sd_config.build_scheduler(n_steps);

    let tokenizer = clip::Tokenizer::create(vocab_file, &sd_config.clip).unwrap();
    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer.encode(prompt).unwrap();
    let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
    let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(clip_device);
    let uncond_tokens = tokenizer.encode("").unwrap();
    let uncond_tokens: Vec<i64> = uncond_tokens.into_iter().map(|x| x as i64).collect();
    let uncond_tokens = Tensor::from_slice(&uncond_tokens)
        .view((1, -1))
        .to(clip_device);

    let no_grad_guard = tch::no_grad_guard();

    println!("Building the Clip transformer.");
    let text_model = sd_config
        .build_clip_transformer(&clip_weights, clip_device)
        .unwrap();
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device);

    println!("Building the autoencoder.");
    let vae = sd_config.build_vae(&vae_weights, vae_device).unwrap();
    println!("Building the unet.");
    let unet = sd_config.build_unet(&unet_weights, unet_device, 4).unwrap();

    let bsize = 1;
    let idx = 0;
    tch::manual_seed(seed + idx);
    let mut latents = Tensor::randn(
        [bsize, 4, sd_config.height / 8, sd_config.width / 8],
        (Kind::Float, unet_device),
    );

    // scale the initial noise by the standard deviation required by the scheduler
    latents *= scheduler.init_noise_sigma();

    for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
        println!("Timestep {timestep_index}/{n_steps}");
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

        let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
        let noise_pred = noise_pred.chunk(2, 0);
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        let noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
        latents = scheduler.step(&noise_pred, timestep, &latents);
    }

    println!(
        "Generating the final image for sample {}/{}.",
        idx + 1,
        num_samples
    );
    let latents = latents.to(vae_device);
    let image = vae.decode(&(&latents / 0.18215));
    let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
    let image = (image * 255.).to_kind(Kind::Uint8);

    let img = &image.squeeze_dim(0).permute([1, 2, 0]);

    let (width, height, _channels) = img.size3().unwrap();
    let mut image_buffer: ImageBuffer<Rgb<u8>, _> = ImageBuffer::new(width as u32, height as u32);

    for w in 0..width {
        for h in 0..height {
            let pred: &Vec<i64> = &img.get(w).get(h).iter::<i64>().unwrap().collect();
            let r = pred[0] as u8;
            let g = pred[1] as u8;
            let b = pred[2] as u8;

            let pixel = Rgb([r, g, b]);

            image_buffer.put_pixel(h as u32, w as u32, pixel);
            // image_buffer.put_pixel(h as u32, (width - 1 - w) as u32, pixel);
        }
    }

    // Create a buffer to store the image data
    let mut buffer: Vec<u8> = Vec::new();

    // Write the image data to the buffer
    image::codecs::png::PngEncoder::new(&mut buffer)
        .encode(
            &image_buffer,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();


    drop(no_grad_guard);

    HttpResponse::Ok()
        .append_header(("content-type","image/png"))
        .body(buffer)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(hello))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}
