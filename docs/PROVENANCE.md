# Dataset provenance and permitted use

## Local inputs inspected

The submission workspace contains two user-provided archives. The audit pipeline inspected them in place without downloading or extracting them:

| Archive | Observed content | Project use |
|---|---|---|
| `A. RFMiD_All_Classes_Dataset.zip` | 3,200 RGB PNGs: 1,920 training, 640 validation, 640 testing; three ground-truth CSVs | RFMiD self-supervised pretraining uses training images only; validation/testing are evaluation-only |
| `B. Disease Grading.zip` | 516 RGB JPEGs: 413 training, 103 testing; disease-grading CSVs | IDRiD ordinal grading and external-transfer evaluation; never RFMiD pretraining |

The local audit report is [audit_report.json](../results/audit/audit_report.json). It records the observed CSV columns, image dimensions/modes, readability, duplicate groups, cross-split duplicates, and CSV matching.

## Source citations

- RFMiD dataset descriptor: [Pachade et al., Data 2021, 6(2), 14](https://www.mdpi.com/2306-5729/6/2/14), which describes 3,200 retinal images and their expert annotations.
- RFMiD challenge/source portal: [RIADD/RFMiD download page](https://riadd.grand-challenge.org/download-all-classes/).
- IDRiD official portal: [IDRiD Grand Challenge home](https://idrid.grand-challenge.org/Home/).
- IDRiD participation and attribution rules: [official Rules page](https://idrid.grand-challenge.org/Rules/).
- IDRiD data descriptor: [Porwal et al., Data 2018, 3(3), 25](https://www.mdpi.com/2306-5729/3/3/25).

## Permitted-use position

The official IDRiD page states that the dataset is publicly available for research through IEEE DataPort and requires appropriate citation; its challenge rules also govern participation and result disclosure. The official RFMiD/RIADD and publication pages should be treated as the source of truth for access and attribution terms.

The supplied ZIP archives do not provide a separate license file in the inspected top-level contents. This repository therefore does not make a stronger redistribution or commercial-use claim. Teams must verify the current provider terms, preserve attribution, follow challenge registration/submission rules, and obtain organizer clarification before redistributing images or derived artifacts. This project code is not legal advice.

