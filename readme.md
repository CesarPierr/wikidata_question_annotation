# Natural Language to Wikidata Query Generation

## Problem Overview

This project addresses a crucial challenge in question answering systems: converting natural language questions into structured Wikidata queries. The specific task involves identifying and tagging entities in questions with their corresponding Wikidata identifiers.

## Task Description

### Input/Output Format

**Input:** Natural language questions (question_raw)  
**Output:** Same questions with properly tagged Wikidata entities (question_tagged)

### Example
**Input:**  
"What is the country for head of state of Justin Trudeau"

**Output:**  
"what is the <<wd:Q6256>> for <<wdt:P35>> of <<wd:Q3099714>>"

### Entity Types
- **wd:** Wikidata entities (e.g., Q6256 for "country")
- **wdt:** Wikidata properties (e.g., P35 for "head of state")
- **ps:** Property statements
- **pq:** Property qualifiers

## Challenges

1. **Entity Recognition**
   - Identifying which parts of the question should be replaced with Wikidata tags
   - Handling multiple entities in a single question
   - Maintaining question structure while replacing entities

2. **Tag Generation**
   - Generating correct Wikidata IDs for identified entities
   - Ensuring proper tag format (<<type:ID>>)
   - Maintaining proper order of tags in complex questions

3. **Semantic Understanding**
   - Understanding question context to generate appropriate tags
   - Distinguishing between different types of entities (wd, wdt, ps, pq)
   - Preserving question meaning while replacing text with tags

## Evaluation Metrics

- **F1 Score:** Primary metric for evaluating model performance
- **Baseline Target:** 25% F1 score

## Dataset Structure

The dataset contains three main splits:
- Training set
- Validation set 
- Test set

Each set contains paired examples of:
- Raw questions (question_raw)
- Tagged questions (question_tagged)

## Solutions

Many solutions have been tested:
- the first one in `bert_base_solution` with a simple bert backbone encoder achieving 25% f1
- the second in `t5_solution` with a t5 decoder and a custom loss with only 17% final f1
- the last in `custom_solution` is a bert_base solution with a custom loss and output type and data augmentation achieving up to 75% f1 