# Text Exploration with BERT

This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for text exploration. It presents a simple approach to tokenization and training on the `c4` dataset and makes predictions for masked tokens within text.

## Setup

Before running the project, ensure the following libraries are installed:

```bash
pip install torch transformers datasets
```

## Usage

The script predicts the possible word(s) for a `[MASK]` token within a given piece of text. Example usage:

```python
input_text = "Chocolate is the best [MASK] treat."
predicted_output = predict(input_text, model, tokenizer)
print("Predicted Output:", predicted_output)
```

This will return the model's prediction for the word or phrase that could fill in the `[MASK]` token.

## Functions

- `predict(input_text, model, tokenizer)`: Makes predictions for the `[MASK]` token within the provided `input_text`.

## Contributing

If you would like to contribute to the project, please open an issue or directly submit a pull request.

