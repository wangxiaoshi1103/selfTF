import torch
import clip
from PIL import Image
import os
import glob
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import pdb

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型
model_path = "../.cache/modelscope/hub/models/AI-ModelScope/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)
# 创建存储结构
embeddings = []

# 遍历图片文件
for img_path in glob.glob('filter/*.jpg'):
    try:
        # 处理图像
        image = Image.open(img_path)
        
        # 生成文本描述（这里用文件名作为示例）
        text_description = os.path.splitext(os.path.basename(img_path))[0]
        inputs = processor(text=[text_description],images=image,return_tensors="pt",padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        image_emb = outputs.image_embeds[0].cpu().numpy()
        text_emb = outputs.text_embeds[0].cpu().numpy()
        # 处理文本
        
        # 存储结果
        embeddings.append({
            "image_embedding": image_emb.astype(np.float32),
            "text_embedding": text_emb.astype(np.float32),
            "text": text_description
        })
        
    except Exception as e:
        print(f"处理文件 {img_path} 时出错: {str(e)}")

# 格式2: PyTorch文件
torch.save(embeddings, "embeddings2.pt")

print(f"处理完成，共处理 {len(embeddings)} 个样本")
