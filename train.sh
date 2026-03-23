source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate base && \
$(conda info --base)/bin/python language_model.py \
    --epochs 15 \
    --batch_size 16 \
    --context 256 \
    --stride 128 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 8 \
    --ffn_dim 2048 \
    --dropout 0.1 \
    --lr 3e-4
