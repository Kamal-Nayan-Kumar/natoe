Overview
A radiologist often begins with short, telegraphic dictation rather than a finished report. Your task is to convert that dictation into a complete structured report by editing a supplied normal template.

For every case, test.csv provides exactly eight input columns: case_id, modality, body_part, study_description, patient_age_band, patient_sex, template_content, and dictation. Candidates generate one report for each case_id. Return FINDINGS and IMPRESSION only.

Core reporting rule
Treat the template as the starting report, not merely as an example.

Route every dictated finding to the matching FINDINGS field.
Replace or modify the corresponding normal statement when the dictation describes an abnormality.
Preserve template statements for routinely visualized regions that were not mentioned in the dictation.
Update the IMPRESSION so it accurately summarizes the important abnormal findings.
Do not add findings unsupported by dictation or template_content. Use modality, body_part, study_description, patient_age_band, and patient_sex only as context; these columns do not supply additional findings.
Preserve the supplied template wording and report order whenever possible.
When the provided template contains an OTHER FINDINGS: field, use it for relevant findings that do not belong in another labelled field. Leave it empty when no such finding applies.
FINDINGS and IMPRESSION
FINDINGS contains the detailed observations organized by anatomy or template field. IMPRESSION is the concise clinical summary: it should report the major abnormal findings and may include a possible diagnosis when that diagnosis is supported by the FINDINGS. It should not repeat every normal statement or introduce new information.

The expected report deliberately remains close to the supplied template. Avoid rewriting unchanged normal statements simply for style. Semantically reasonable paraphrases may still receive an edit penalty because the leaderboard measures template-edit fidelity.

Data notice: This benchmark uses de-identified radiology text. case_id is random; exact ages, patient identifiers, accession numbers and study UIDs are not provided. All dates have been removed from the text. The data must not be used for clinical decisions.

Input columns
Column	Meaning
case_id	Random UUID used to match test cases to submission rows.
modality	Imaging modality: XRAY, MRI, CT, or USG.
body_part	Anatomical region examined.
study_description	Name or description of the ordered examination.
patient_age_band	Five-year age band, such as 45-49; ages 90 and above are 90+.
patient_sex	Patient sex recorded as male or female.
template_content	Normal FINDINGS fields and IMPRESSION that must be edited.
dictation	Transcribed radiologist dictation containing the findings to incorporate.
train.csv contains these eight input columns plus the target report. test.csv contains exactly the eight input columns above and does not contain report. sample_submission.csv contains exactly case_id and report.

No clinical history, prior study or image is provided. Do not infer unsupported facts.

Worked chest radiograph example
Dictation

mild right basilar opacity, small right pleural effusion
Normal template

FINDINGS:
SUPPORT DEVICES: None.
CARDIOMEDIASTINAL SILHOUETTE: Within normal size limits.
LUNGS: No focal airspace opacity or pulmonary edema.
PLEURA: No pleural effusion or pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
No acute cardiopulmonary abnormality.
Expected report

FINDINGS:
SUPPORT DEVICES: None.
CARDIOMEDIASTINAL SILHOUETTE: Within normal size limits.
LUNGS: Mild right basilar airspace opacity. No pulmonary edema.
PLEURA: Small right pleural effusion. No pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
Mild right basilar airspace opacity and small right pleural effusion.
The lung and pleural statements changed because those abnormalities were dictated. The support devices, cardiomediastinal and osseous fields remained normal because nothing in the input required changing them. The opacity must not be placed under PLEURA, and the effusion must not be placed under LUNGS.

Practical starting approaches
There are no restrictions on the model family or implementation method. Reasonable starting points include:

Prompt engineering: give a capable language model modality, body_part, study_description, patient_age_band, patient_sex, template_content, dictation, and explicit field-preservation rules. Ask it to return only the completed report.
Structured generation: extract findings into a schema, route them to template fields, then render the final report deterministically.
Synthetic-data alignment: generate paired dictation/template/report examples and fine-tune or post-train a smaller model to perform template editing.
Hybrid or agentic pipeline: combine extraction, validation and generation stages, optionally using a hosted model API.
Whatever method you choose, verify negation, laterality, measurements, field routing, unsupported additions and preservation of untouched normal fields.

Submission and hiring review
Upload the completed submission.csv through Kaggle's File Upload option. You may generate the reports outside Kaggle, including with external model APIs. Each report must contain only FINDINGS and IMPRESSION.

A complete hiring-challenge entry must also include the automated pipeline as a Kaggle Notebook:

Create or upload a Kaggle Notebook containing the complete pipeline code. The notebook does not need to execute on Kaggle.
Save a notebook version and share the private notebook with Natoe AI Dev (natoeaidev).
Paste the notebook URL into the Submission Description of your final CSV submission.
A leaderboard submission without an accessible pipeline notebook is incomplete. Do not include API keys; use environment-variable or secret placeholders. The submitted code must reproduce the uploaded CSV without manual case-level editing.

The Kaggle leaderboard uses only RES - Radiology Edit Score, where lower is better. RES measures minimal template-edit fidelity, not full semantic or clinical equivalence. The hiring team may separately review clinical faithfulness, unsupported content, omissions, routing, reproducibility, and system design; this review does not change the Kaggle leaderboard score.

Start

Sep 3, 2026
Close

Sep 8, 2026
Description
At a glance
This private hiring challenge is hosted by Natoe.ai.

File	Rows	Columns	Purpose
train.csv	636	9	Inputs plus the reference report
test.csv	132	8	Inputs only; generate one report per case
sample_submission.csv	132	2	Required submission structure
Each test case provides exactly case_id, modality, body_part, study_description, patient_age_band, patient_sex, template_content, and dictation. No additional DICOM or study tags are supplied.

Generate a complete report containing FINDINGS and IMPRESSION. The leaderboard uses the field-aware Radiology Edit Score described below; lower is better.

Submit a CSV with exactly case_id and report. For the required pipeline review, follow the Kaggle Notebook workflow in the Overview: share a saved private notebook with natoeaidev and include its URL in the final submission description.

Evaluation
Submissions are ranked by RES - Radiology Edit Score. Lower values are better. A prediction that exactly matches the reference report after normalization receives 0.0.

Required report structure
Every submitted report must contain exactly two top-level sections:

FINDINGS:
<FINDINGS fields>

IMPRESSION:
<concise summary>
Within FINDINGS, retain the supplied template's field labels, such as LUNGS:, PLEURA: or BONES:. The scorer uses these labels to align corresponding fields. A finding placed under the wrong label is treated as missing from its expected field, while its content under the unexpected label is penalized as extra content. Follow the template's field order even though alignment is performed by label.

IMPRESSION is compared as one complete section.

Text normalization
Before edit distance is calculated, reference and submitted text are normalized consistently:

Unicode is normalized and text is converted to lowercase.
Punctuation is ignored except that a leading + or - attached directly to a number is preserved as part of a signed measurement.
Leading list markers such as 1., 2), -, * and • are treated as formatting and removed before scoring.
Hyphens between letters are removed, so air-space and airspace are equivalent.
Letter-number boundaries are separated into tokens.
Common unit spellings are standardized, such as millimeters to mm and centimeters to cm.
Word order is preserved. Reordering otherwise correct words can therefore increase RES.
Weighted word edit distance
Each field is compared using ordered word-level Levenshtein distance: the minimum weighted cost of inserting, deleting or substituting tokens.

Token type	Weight	Examples
Clinically critical words, measurements and units	4.00	negation, laterality, severity, acuity, numbers, mm, cm
Other content words	2.00	anatomy and descriptive findings
Common function words	0.25	the, and, of, with
The edit cost is divided by the larger total token weight of the reference or submitted text and capped at 1. Therefore each word-edit score lies between 0 and 1.

Field-aware FINDINGS score
For each reference FINDINGS field, the scorer first compares the reference field with the same field in template_content:

A field changed from the normal template receives field weight 3.
A field unchanged from the normal template receives field weight 1.
A missing expected field receives the edit penalty for comparing its reference content with an empty value.
Unexpected fields and unsupported unlabelled FINDINGS content receive additional penalties.
The FINDINGS score is:

F = sum(field_weight × field_word_edit) / sum(field_weight)
This gives more influence to fields containing the abnormalities that had to be incorporated while still rewarding preservation of untouched normal fields.

Final case and leaderboard score
The IMPRESSION score I is the weighted word edit distance over the complete IMPRESSION section.

RES_case = 0.65 × F + 0.35 × I
Leaderboard_RES = mean(RES_case across scored cases)
Lower is better. RES evaluates ordered template-edit fidelity; it is not a general semantic-equivalence metric. Unsupported additions, omissions, incorrect routing and unnecessary rewriting normally increase the score.

Submission file
Submit one CSV containing exactly two columns:

case_id,report
Include every case_id from test.csv exactly once. The report cell may contain line breaks and must be quoted correctly by the CSV writer. Do not include template_content, dictation, Usage or an index column.

Example:

case_id,report
123e4567-e89b-12d3-a456-426614174000,"FINDINGS:
LUNGS: No focal airspace opacity.
PLEURA: No pleural effusion.

IMPRESSION:
No acute cardiopulmonary abnormality."
Competition Host
Natoe AI Dev

Prizes & Awards
Kudos

Does not award Points or Medals

Participation
507 Entrants

262 Participants

262 Teams

900 Submissions

Tags
Custom Metric
Table of Contents
----------------
Dataset:
Dataset Description
Files
train.csv
636 labeled rows and 9 columns. It contains the eight input columns plus the target report.

test.csv
132 unlabeled rows and 8 columns. It contains:

case_id
modality
body_part
study_description
patient_age_band
patient_sex
template_content
dictation
Generate one final report for every case_id.

sample_submission.csv
132 rows and 2 columns: case_id and report. Preserve every test case_id exactly once and place the complete generated FINDINGS and IMPRESSION text in report.

Column definitions
case_id: random UUID used only for row matching.
modality: imaging modality.
body_part: examined body region.
study_description: study or examination description.
patient_age_band: broad de-identified age group.
patient_sex: de-identified sex value.
template_content: normal structured report used as the starting point.
dictation: telegraphic radiologist observations to incorporate.
report: target final report; present only in train.csv.
No additional DICOM or study metadata is provided. The text is de-identified, case IDs are random, exact ages and direct identifiers are absent, and all dates have been removed. Do not use this data for clinical decisions.

