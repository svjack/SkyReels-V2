import os
import gc
import time
import random
import torch
import imageio
from diffusers.utils import load_image
from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer, resizecrop

# ---------------------
# 全局初始化部分（只执行一次）
# ---------------------

is_shared_ui = True
model_id = download_model("Skywork/SkyReels-V2-DF-1.3B-540P") if is_shared_ui else None

# 预设分辨率参数
RESOLUTION_CONFIG = {
    "540P": (544, 960),
    "720P": (720, 1280)
}

# 负向提示词（固定）
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# 初始化 pipeline（只初始化一次）
pipe = DiffusionForcingPipeline(
    model_id,
    dit_path=model_id,
    device=torch.device("cuda"),
    weight_dtype=torch.bfloat16,
    use_usp=False,
    offload=True,
)
# ---------------------
# 函数定义部分
# ---------------------


def generate_diffusion_forced_video(
    prompt,
    image=None,
    target_length="10",
    model_id="Skywork/SkyReels-V2-DF-1.3B-540P",
    resolution="540P",
    num_frames=257,
    ar_step=0,
    causal_attention=False,
    causal_block_size=1,
    base_num_frames=97,
    overlap_history=17,
    addnoise_condition=20,
    guidance_scale=6.0,
    shift=8.0,
    inference_steps=30,
    use_usp=False,
    offload=True,
    fps=24,
    seed=None,
    prompt_enhancer=False,
    teacache=True,
    teacache_thresh=0.2,
    use_ret_steps=True,
):
    """
    使用已初始化的 pipeline 进行视频生成，仅需传入动态参数
    """
    # 获取分辨率
    if resolution not in RESOLUTION_CONFIG:
        raise ValueError(f"Invalid resolution: {resolution}")
    height, width = RESOLUTION_CONFIG[resolution]

    # 设置种子
    if seed is None:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))

    # 检查长视频参数
    if num_frames > base_num_frames and overlap_history is None:
        raise ValueError("Specify `overlap_history` for long video generation. Try 17 or 37.")
    if addnoise_condition > 60:
        print("Warning: Large `addnoise_condition` may reduce consistency. Recommended: 20.")

    # 图像处理
    pil_image = None
    if image is not None:
        pil_image = load_image(image).convert("RGB")
        image_width, image_height = pil_image.size
        if image_height > image_width:
            height, width = width, height
        pil_image = resizecrop(pil_image, height, width)

    # 提示词增强
    prompt_input = prompt
    if prompt_enhancer and pil_image is None:
        enhancer = PromptEnhancer()
        prompt_input = enhancer(prompt_input)
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    # TeaCache 初始化（如启用）
    if teacache:
        if ar_step > 0:
            num_steps = (
                inference_steps + (((base_num_frames - 1) // 4 + 1) // causal_block_size - 1) * ar_step
            )
        else:
            num_steps = inference_steps
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=num_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
            ckpt_dir=model_id,
        )

    # 是否开启因果注意力
    if causal_attention:
        pipe.transformer.set_ar_attention(causal_block_size)

    # 生成视频
    with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            image=pil_image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            overlap_history=overlap_history,
            addnoise_condition=addnoise_condition,
            base_num_frames=base_num_frames,
            ar_step=ar_step,
            causal_block_size=causal_block_size,
            fps=fps,
        )[0]

    # 保存视频
    os.makedirs("gradio_df_videos", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"gradio_df_videos/{prompt[:50].replace('/', '')}_{seed}_{timestamp}.mp4"
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    return output_path

import os
from datasets import load_dataset
from PIL import Image
from diffusers.utils import load_image

# 加载数据集
dataset = load_dataset("svjack/Mavuika_PosterCraft_Product_Posters_WAV")["train"]

# 初始化输出目录
output_dir = "Mavuika_generated_videos"
os.makedirs(output_dir, exist_ok=True)

# 循环遍历数据集
for idx, item in enumerate(dataset):
    try:
        # 获取图像和提示词
        pil_image = item["postercraft_image"]
        prompt = item["final_prompt"]

        # 保存原始图片为临时文件供 generate_diffusion_forced_video 使用
        temp_input_path = f"temp_input_{idx:04d}.png"
        pil_image.resize((544, 960)).save(temp_input_path)

        # 调用视频生成函数
        video_path = generate_diffusion_forced_video(
            prompt=prompt,
            image=temp_input_path,
            target_length="4",  # 可选参数，实际使用 height/width 控制长度
            model_id="Skywork/SkyReels-V2-DF-1.3B-540P",
            resolution="540P",
            num_frames=97,
            ar_step=0,
            causal_attention=False,
            causal_block_size=1,
            base_num_frames=97,
            overlap_history=3,
            addnoise_condition=0,
            guidance_scale=6,
            shift=8,
            inference_steps=30,
            use_usp=False,
            offload=True,
            fps=24,
            seed=None,
            prompt_enhancer=False,
            teacache=True,
            teacache_thresh=0.2,
            use_ret_steps=True,
        )

        # 构建输出路径
        output_video_path = os.path.join(output_dir, f"{idx:04d}.mp4")
        output_txt_path = os.path.join(output_dir, f"{idx:04d}.txt")

        # 移动视频文件到输出目录
        os.rename(video_path, output_video_path)

        # 保存 prompt 到 .txt 文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        print(f"✅ 已生成并保存：{output_video_path}")

    except Exception as e:
        print(f"❌ 处理第 {idx} 张图片时出错: {e}")