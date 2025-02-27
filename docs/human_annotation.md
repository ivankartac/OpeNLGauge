# Human Annotation Guidelines

Please help us evaluate LLM-based NLG evaluators.  

We’re using the RotoWire dataset (basketball domain) from [Thomson & Reiter (2020)](https://aclanthology.org/2020.inlg-1.22/). You will be given an input data, generated output, and a list of *faithfulness* errors as identified by 3 different LLM evaluators. For the purpose of this task, we define *faithfulness* as extent to which the information in the generated text is supported by the input data.

## Task Description

You will use a spreadsheet with a list of faithfulness errors in NLG outputs as identified by 3 different LLM evaluators. Input data and generated outputs are loaded in [[Factgenie](https://github.com/ufal/factgenie) URL].

Please add your initials in the **annotator ID** column (rightmost) for one data instance, i.e. one NLG system output, coming from all 3 evaluators (all items with the same value in the leftmost ID column). Note that annotations from different evaluators are marked by changing colors of the ID column. The names of the LLM evaluators are anonymized.

Please go through all identified errors and corresponding explanations for the given data instance, check them against the Factgenie data and the full text view, and mark the following categories:

### 1. Error span OK?

* **Error**: The error span contains an actual error  
* **Not an error**: The error span doesn’t contain an error, the text is fine  
* **No span given**: No explicit error span given  
    * This includes cases of “the data mentions X”, which are additionally marked with the “implicit span” flag  
* **Hallucination**: Span is completely hallucinated (i.e. given but non-existent in the data)  

### 2. Explanation OK?

* **Correct**: The explanation is correct and includes most important points  
* **Partially correct**: Part of the explanation is correct, but it also includes an incorrect or nonsensical statement  
* **Incomplete**: The explanation is correct, but: 
    * misses a part of the reasoning (e.g. it says a statement is false, but doesn’t mention why)
    * mentions only some of the errors that are in the span, while there are more that should also be addressed  
* **Vague**: The explanation is too vague to explain anything  
* **Incorrect**: The explanation does not make sense, or the reasoning is incorrect  
* **Not an error**: The error span and the explanation do not address an error in the generated text  

### 3. Additional flags for the span

*Note: Can be none or multiple*

* **Too strict**: The error is technically an error, but the metric is nitpicking
* **Not aspect related**: Not a faithfulness problem (e.g. text is redundant or repetitive, ungrammatical, etc.)  
* **Repeated**: Error span reported repeatedly by the same system  
* **Hard to parse**: The indicated error span is hard to parse. It’s not just the text itself, it’s more spans in fact, or it has “...” in the middle. This doesn’t apply to whitespace/tokenization differences.  
* **Implicit span**: When the span is not given, but there’s an implicit mention of it (“the data mentions X”)

### 4. Comments

Anything you’re unsure about or want to point out.
