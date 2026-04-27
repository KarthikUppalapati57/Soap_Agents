# MKG Validation — Medical Knowledge Graph

## Overview
Validates medical terms in generated SOAP notes using UMLS API.

## Tools Used
- scispaCy → extracts medical terms from SOAP note
- UMLS API → validates each term against medical databases

## What UMLS Checks
- SNOMED-CT → symptoms and diagnoses
- RxNorm → medications and drugs
- MeSH → medical vocabulary
- ICD-10 → diagnosis codes
- LOINC → lab tests

## Scoring
MKG Score = Valid Terms / Total Terms

## References
- UMLS: https://uts.nlm.nih.gov