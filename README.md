Brief Summary
Timing violations in RTL designs delay project execution due to slow synthesis processes. This project develops an AI model to predict combinational logic depth for signals in RTL modules. The approach involves dataset creation, feature engineering, machine learning model selection, training, and evaluation. By accurately estimating logic depth pre-synthesis, the proposed solution accelerates timing analysis and reduces design iterations.

Problem Statement
Modern semiconductor designs require rigorous timing analysis to ensure functional correctness and optimal performance. Timing violations arise when the combinational logic depth exceeds the permissible threshold for a given clock cycle. Current methodologies rely on full synthesis reports, which are computationally expensive. This project aims to develop a machine learning model that predicts combinational logic depth in behavioral RTL without requiring full synthesis. This solution benefits hardware engineers by reducing analysis time and improving design efficiency.

The Approach Used to Generate the Algorithm

Dataset Creation: Gather RTL design files and corresponding synthesis reports to extract logic depth information for key signals.

Feature Engineering: Extract key features affecting logic depth, such as fan-in, fan-out, logic gate count, and signal path length.

ML Model Selection: Compare multiple models like Decision Trees, Random Forests, and Neural Networks for accuracy and runtime efficiency.

Training: Use supervised learning to train the model on annotated datasets with known logic depths.

Evaluation: Split dataset into training and testing sets to validate prediction accuracy against ground truth from synthesis reports.

Proof of Correctness

The model's predictions are compared with actual logic depth values extracted from synthesis tools.

Cross-validation techniques ensure robustness.

Performance is measured using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

The AI model is iteratively fine-tuned to improve prediction accuracy based on evaluation results.

Complexity Analysis

Preprocessing Complexity: O(n) for extracting features from RTL files.

Training Complexity: Depends on the ML model used; Decision Trees have O(n log n), whereas Neural Networks may have O(n^2) complexity.

Inference Complexity: O(1) per signal, making real-time predictions feasible.

Alternatives Considered

Rule-Based Heuristics: Manually defined logic depth estimation rules, but lacked flexibility and accuracy.

Graph-Based Algorithms: Using graph traversal to estimate depth but required synthesis-like preprocessing.

Hybrid ML + Heuristic Models: Combining rule-based methods with ML, but found ML models alone to be more efficient.

References and Appendices

IEEE papers on timing analysis and logic depth estimation.

Open-source EDA tools like Yosys for dataset generation.

Public datasets of synthesized RTL designs with annotated logic depths.

Diagrams illustrating combinational paths and feature extraction methodology.
