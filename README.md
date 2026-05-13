Steps:
1. Load image                        V
2. Resize to 224*224                 V
3. Seperate to patches               V
4. Flatten the patches               V
5. Add positional encoding           V
6. Add class token                   V
7. modify the code to use dataloader instead of single image
8. Input to Transformer
9. Use MLP head for multi class 
10. Add loss function 
11. add train function
12. load dataset
13. test