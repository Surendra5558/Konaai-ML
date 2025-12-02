# Release Notes
## 2nd November 2023
1. Now supports three JWT authentication methods
    a. SecretKey - Change CertificateType to SecretKey and set value of SecretKey. Not recommended, added for backward compatability
    b. PemKeys - Change CertificateType to PemKeys and upload public and private key. pem files.
    c. KeyVault - Change CertificateType to KeyVault and update certificate name and AzureKeyVaultName.
2. Added new security headers in API based on penetration testing findings.
3. Added Expense Receipts OCR Module.
4. OCR Evaluation file alternate path set to assets directory
5. Added Contract OCR Module.

## 14th November 2023
6. Added KonaAIStreamlit service on port 8501 in deployment script
7. Fixed OCR keyword highlighter, actual word and lemmatized keyword difference bug.

## 20th November 2023
1. Fixed the issue related to case and order of keywords
2. Updated some exception messages to error messages.

## 22nd November 2023
1. Fixed .msg and .eml file extraction logic

## 4th December 2023
1. Invoice Anomaly dashboarded delivered
2. Single UDM Common Model for P2P refactored

## 5th December 2023
1. Changed python version to 3.11
2. Added an empty document in OCR output JSON when no document is extracted to fix bug related to enable addition of headers and items.

## 12th December 2023
1. Removed 90 word limit from keyword evaluations. This limit was introduced to avoid invalid text to be evaluated in case of tabular document such as invoices. Based on customer input, removed it.

## 13th December 2023
1. Fixed bug related to case of documents for OCR full text extraction. Earlier OCR failed for documents that had extension in upper case.
2. Updated streamlit interface for following:
    a. Instance creation error handling updated.
    b. Bearer token check text area UI improved.
    c. Removed active instance requirement for a few checks in health check section.

## 3rd January 2023
1. Removed performance monitoring.
2. Removed multiple log files and consolidated to single log file.
3. Changed all exception logging to error logging.
4. Updated validation message for OCR date validation.
5. Fixed bugs for anomaly features to cover all rows in output irrespective of feature output numbers.
6. Added a condition in anomaly features to run only relevant features for the corresponding submodule.
7. Implemented a threshold condition for anomaly features to consider only above threshold as anomaly.
8. Fixed bug related to p2p data reduction that ignored non credit/debit values.

## 16th February 2024
1. Fixed instance_setting issue adding task_queue to resolve it.
2. Fixed bug related to streamlit frontend experimental_get_query_params issue by changing query_params.

## 22nd February 2024
1. Fixed Bug 9076 - Text extraction failing for embedded images inside a pdf documents.
2. Updated supported file formats for OCR full document extraction. TIF and TIFF file formats no longer supported.
3. Refactored OCR extraction code to avoid JSON load issues.
4. Added automated test cases for text extraction from documents.
5. Added Anomaly risk scoring button in streamlit

## 28th February 2024
1. Bug 9158 - OCR: Email file upload highlighter bug fixed.

## 7th March 2024
1. Added p2p single UDM risk scoring

## 27th March 2024
1. Fixed bug related to special characters in OCR document URL. Bug 9552

## 03rd April 2024
1. Fix for Bug 9552 was causing another challenge. Since URL are encoded by applicatino, file name to be downloaded got changed hence failing for Azure storage. Unquoted the file name.

## 15th April 2024
1. Added AutoML capability

## 16th April 2024
1. Fixed code security vulnerability issues

## 22nd April 2024
1. Added OCR settings and keyword API for frontend.

## 17th May 2024
1. Updated AutoML question to show only active and mandatory questions
2. Enabled temporary files cleaning on periodic basis
3. Sending task result with status for task api
4. Enabled settings encryption

## 22nd May 2024
1. Fixed instance creation form issue. Bug ID 11456

## 03rd June 2024
1. Enabled AutoML for all possible modules and submodules

## 25th June 2024
1. Added T&E Expenses Anomaly Pipeline

## 28th June 2024
1. Added Task Management Page in streamlit

## 3rd July 2024
1. Added redis.conf file with reddis configration.

## 23rd July 2024
1. Added Normalization in Anomaly Pipeline. [3.6.0]
2. Fixed mlflow tracking URI change issues that happens when new version is deployed. Bug 517. [3.6.1]

## 30th July 2024
1. Auto excution of risk scoring procedure for all ml pipeline. [3.7.0]
2. Enabled AutoML Run settings with pipeline [3.7.1]

## 13th August 2024 [3.8.0]
1. Replaced Redis as a celery broker with RabbitMQ
2. Replaced Redis as celery backend with MSSQL
3. Fixed AutoML Run Settings widget draw error in streamlit
4. Added NLTK as part of package. This will avoid downloading nltk.

## 26th August 2024 [3.9.0]
1. Removed celery mingling since we are using single instance
2. Updated dask version

## 2nd September 2024 [3.10.0]
1. Enabled all categorical columns to be used as feature in AutoML
2. Removed file traces from logs and Improved logging.
3. Enabled message cancellation from RabbitMQ Queue using streamlit to cancel tasks.

## 6th September 2024 [3.10.0]
1. Updated AutoML feature importance
2. Updated confusion matrix key names for clarity
3. Added last prediction details for AutoML verification
4. Added AutoML Risk Scoring SQL as part of risk scoring task
5. Added AutoML risk scoring validation

## 17th September 2024 [3.10.1]
1. Fixed bug 946 - Quantity was not extracted for OCR
2. Optimized expense anomaly
3. Optimized Invoices anomaly
4. Optimized Payments anomaly
5. Optimized Purchase Orders anomaly

## 8th October 2024 [3.10.1]
1. Updated anomaly count features.
2. Optimized AutoML processing.

## 9th October 2024 [3.10.1]
1. Integrated Risk Scoring for TE Expenses Anomaly within ML Code
2. Added Benford Anomaly for TE

## 10th October 2024 [3.10.1]
1. Added Benford Anomaly for Invoices
2. Added Benford Anomaly for P2P Payments
3. Added Benford Anomaly for Purchase Orders

## 14th October 2024 [3.10.1]
1. Integrated Risk Scoring for P2P Anomaly within ML Code

## 18th October 2024 [3.10.1]
1. Integrated AutoML Risk Scoring

## 22nd October 2024 [3.10.2]
1. Enhanced log viewer exeperience
2. Fixed dask compute bug. Bug ID 2415
3. Enabled default risk scoring for Anomaly and AutoML when triggered via API
4. Changed Benford anomaly from binary to continuous scale

## 4th November 2024 [3.10.3]
1. Fixed Bug 2707 for Invoice anomaly. Changed other anomaly features to avoid similar issues.

## 11th November 2024 [3.11.0]
1. Updated AutoML Task output to include experiment details for AutoML Audit capture.

## 13th November 2024 [3.11.1]
1. Fixed bug 3265 - Dask to_datetime conversion caused dask dataframe data types misalignment.

## 22nd November 2024 [3.12.0]
1. User Story 2454 - To update single UDM risk scoring and to align with AUTO ML

## 26th November 2024 [3.12.1]
1. Bug 3682: Task status api returned error when task status could not be found. Added graceful response.

## 2nd December 2024 [3.13.0]
1. Added amount column features to AutoML
2. Added anomaly features to AutoML
3. Enabled model hyperparameter tuning
4. Enabled decision threshold turning
5. Enabled rare label reduction for categorical columns
6. Optimized Benford anomaly perforamnce

## 11th December 2024 [3.14.0]
1. Added email notifications to Anomaly and AutoML pipelines

## 27th January 2025 [3.14.3]
1. Bug 7072 : AutoML Validation error was not sent to application.
2. Bug 7305 : Celery task restarts after timeout even after completing the task.
3. Bug 7461 : T&E Anomaly Pipeline notification email subject is incorrect
4. Added last prediction date API endpoint
5. Added Risk Scoring API endpoints for Anomaly, Single UDM and AutoML Risk Scoring.

## 25th February 2025 [3.15.0]
1. Replace linear correlation based feature selection with mutual information based feature selection [US 10882]
2. Updated rare labels to exclude rare labels present only in no concern records [US 10883]
3. Fixed Archived table name issue [Bug 10884]
4. Fixed invoice anomaly count issue happening due to missing dates [Bug 10885]

## 15th May 2025 [3.15.1]
1. Updated primary key data type for MLS TaskSet table for celery.
2. Updated code to handle SMTP missing configuration.

## 16th May 2025 [3.15.2]
1. Fixed AutoML weight query for contribution percentage.

## 21st May 2025 [3.15.3]
1. Improved task status assessment to avoid false report.

## 13th May 2025 [3.16.0]
1. Updated Goldeb DB column changes for Anomaly features
2. Updated Pattern ID names for P2P anomaly patterns
3. Removed anomaly risk scoring process in tasks, risk scoring for anomaly should be run via stored procedures by DB team due to risk scoring logic change.
4. Temp files cleanup enabled after every task completion.
5. Anomaly features where we are calculating date differences, limited the future date to tomorrow and past date to max 25 years ago.
6. Updated LOF logic for improved results.
7. Fixed dask datatype inference logic
8. Added high cardenality feature filter by proportionality
9. Added Tamplate_Id for Questionnaire, ML_API and ML console
10. Added SMTP relay functionality
11. Exclude Intercompany transactions for P2P invoice anoamly data.


# Known Issues
1.	In case the dependency installation fails due to build process on windows, try installing https://visualstudio.microsoft.com/visual-cpp-build-tools/.

# RabbitMQ Setup
- Installation
https://www.onlinetutorialspoint.com/windows/how-to-install-rabbitmq-on-windows-10.html

- SSL Setup
https://help.matrix42.com/010_SUEM/010_UUX_for_Secure_UEM/090KnowledgeBase/Securing_RabbitMQ_with_TLS%2F%2FSSL

- Troubleshooting for SSL
In case you find SSL issue check for erling cookie value in following files
    - C:\Windows\System32\config\systemprofile\.erlang.cookie
    - C:\Users\<Username>\.erlang.cookie

    The value should be same in both the files.
- Also make sure that rabbitmq.conf is also present in C:\Users\<user>\AppData\Roaming\RabbitMQ
