DL Hyperparameter Tuning
1. Start Small (Baseline Model)
    - 1 hidden layer, small number of neurons
    - Default activation: ReLU
    - Optimizer: Adam, lr=0.001
    - IF train & val both poor → underfitting → go to Step 2

2. Increase Nodes in Hidden Layer
    - Gradually increase neurons in existing layers
    - IF train improves and val improves → nodes sufficient → Step 3
    - IF gap starts widening → possible overfit → stop increasing nodes

3. Increase Number of Layers
    - Add layers one at a time
    - IF val performance plateaus or gap widens → stop adding layers

4. Train Longer (Increase Epochs)
    - Let model fully converge
    - IF val loss stops improving → check Step 7 (overfit mitigation)
    - IF gap small → continue, consider adjusting batch size or learning rate

5. Change Activation Function
    - e.g., ReLU → Leaky ReLU / ELU
    - IF training plateaus, gradients vanish, or dead neurons appear → change activation
    - Step 5b: Fine-tune optimizer or learning rate after activation change

6. Increase Capacity Further (Optional)
    - Only if underfit persists after previous steps
    - IF train loss << val loss → overfitting detected → Step 7

7. Apply Dropout
    - First overfit mitigation method
    - IF train >> val gap remains → Step 8

8. Apply Data Augmentation
    - Especially for small datasets or anomalies
    - IF val improves and gap reduces → Step 9

9. Apply L2 Regularization (Weight Decay)
    - Stabilize training after dropout / augmentation
    - IF gap still exists or weights oscillate → Step 10

10. Early Stopping
    - Stop training if val loss stops improving for N epochs
    - Prevents late overfitting

Steps 2 → 3 → 4 → 5 → 6 focus on underfitting resolution
Steps 7 → 8 → 9 → 10 focus on overfitting mitigation
Some steps are conditional loops, e.g., if overfitting appears after Step 6, revisit Step 4 (train longer with lower LR) or Step 5b (optimizer tweak)
