# Setup
Date: 2025-11-27  
## HGU HPC
    conda create -y --name scgenept python=3.10
    
    source activate scgenept
    
    conda install jupyter
    
    cd scGenePT/
    
    pip install -r requirements.txt

torch 2.1.2+cu121
pip install flash-attn --no-build-isolation
위 코드로 설치 안 돼서 아래 코드로 설치

    pip install flash-attn==1.0.4 --no-build-isolation
    
    pip install scgpt

torch와 torchtext 버전 충돌로 torch 버전 낮춤
    
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"

torchtext 설치 시 torch가 최신 버전으로 업데이트 되어 --no-deps 적용
    
    pip install --no-cache-dir --no-deps "torchtext==0.18.0"
    
    pip install torch_geometric

    pip install ipykernel

***
    pip show scgpt
