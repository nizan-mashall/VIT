Steps:
1. Load image                        V
2. Resize to 224*224                 V
3. Seperate to patches               V
4. Flatten the patches               V
5. Add positional encoding           V
6. Add class token                   V
7. modify the code to use dataloader instead of single image      V
8. Input to Transformer                                           V
9. Use MLP head for multi class                                   V 
10. Add loss function                                             V
11. add train function                                            V
12. load dataset                                                  V
13. test                                                          V

1. run this command after enter the ngc to enter the gpu
srun --pty --nodes=1 --gres=gpu:1 /bin/bash

2. system properties
✅ Enroot 3.1.1 is available
✅ A100 GPU (40GB) — very powerful
✅ CUDA 11.4
✅ Python 3.8.10

3. make req.text file

4. 