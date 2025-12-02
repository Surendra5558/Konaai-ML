SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET QUOTED_IDENTIFIER ON;
SET NOCOUNT ON;
SET ANSI_NULLS ON;

-- ----------------------------------------------------------
-- Step 1
-- Create Required columns in the submodule UDM Table
-- ----------------------------------------------------------
-- Check and add the prediction column if it doesn't exist
IF NOT EXISTS (
		SELECT 1
		FROM sys.columns
		WHERE Name = N'{PredictionColumnName}'
			AND Object_ID = Object_ID(N'{UDM_TABLE_NAME}')
	)
	BEGIN
		/* tsqllint-disable error invalid-syntax */
		ALTER TABLE {UDM_TABLE_NAME} ADD [{PredictionColumnName}] INT NOT NULL DEFAULT 0;
		/* tsqllint-enable error invalid-syntax */

	END
ELSE
	BEGIN
		PRINT 'Columns already exist, skipping creation.'
	END


-- Check and add the {ML_Risk_Score_Column} column if it doesn't exist
IF NOT EXISTS (
		SELECT 1
		FROM sys.columns
		WHERE Name = N'{ML_Risk_Score_Column}'
			AND Object_ID = Object_ID(N'{UDM_TABLE_NAME}')
	)
	BEGIN
		/* tsqllint-disable error invalid-syntax */
		ALTER TABLE {UDM_TABLE_NAME} ADD [{ML_Risk_Score_Column}] FLOAT NOT NULL DEFAULT 0;
		/* tsqllint-enable error invalid-syntax */

	END
ELSE
	BEGIN
		PRINT 'Columns already exist, skipping creation.'
	END

-- ----------------------------------------------------------
-- Step 2
-- Run Risk Scoring Stored Procedure
-- ----------------------------------------------------------
-- Update Prediction Column and Risk Score in UDM Table
DECLARE @UDM_TABLE_NAME NVARCHAR(255) = N'{UDM_TABLE_NAME}';
DECLARE @PredictionTableName NVARCHAR(MAX) = N'{PredictionTableName}';
DECLARE @PredictionColumnName NVARCHAR(MAX) = N'[{PredictionColumnName}]';
DECLARE @PredictionProbabilityColumnName NVARCHAR(MAX) = N'[{PredictionProbabilityColumnName}]';
DECLARE @StageTableName NVARCHAR(MAX) = CONCAT(REPLACE(REPLACE(@UDM_TABLE_NAME,'[',''),']',''),'_STAGE');
DECLARE @SQL NVARCHAR(MAX);

BEGIN TRY
    SELECT @SQL = N'
    UPDATE ' + @UDM_TABLE_NAME + N'
    SET
        [{PredictionColumnName}] = y.' + @PredictionColumnName + N',
        [{ML_Risk_Score_Column}] = CASE
            WHEN y.' + @PredictionColumnName + N' = 1 THEN y.' + @PredictionProbabilityColumnName + N' * 100
            ELSE  0
        END
    FROM ' + @UDM_TABLE_NAME + N' AS x
    INNER JOIN ' + @PredictionTableName + N' AS y
    ON x.[SPT_RowID] = y.[SPT_RowID];';
    EXEC sp_executesql @SQL;

	SELECT @SQL = N'
    UPDATE ' + @StageTableName + N'
    SET
        [{PredictionColumnName}] = y.' + @PredictionColumnName + N',
        [{ML_Risk_Score_Column}] = CASE
            WHEN y.' + @PredictionColumnName + N' = 1 THEN y.' + @PredictionProbabilityColumnName + N' * 100
            ELSE  0
        END
    FROM ' + @StageTableName + N' AS x
    INNER JOIN ' + @PredictionTableName + N' AS y
    ON x.[SPT_RowID] = y.[SPT_RowID];';
    EXEC sp_executesql @SQL;
END TRY

----------------------------------------------------------------------------------------------------
-- check error log

BEGIN CATCH -- Error Details
	INSERT INTO [App].[ErrorLog] (
		[ProcedureName]
		, [ErrorMessage]
		, [ErrorLine]
		, [ErrorSeverity]
		, [ErrorState]
		, [ErrorNumber]
		, [EntitySource]
		, [EntityType]
	)
	VALUES (
		OBJECT_NAME(@@PROCID)
		, ERROR_MESSAGE()
		, ERROR_LINE()
		, ERROR_SEVERITY()
		, ERROR_STATE()
		, ERROR_NUMBER()
		, 'EntitySource'
		,
		-- Replace with actual value
		'EntityType' -- Replace with actual value
	);
END CATCH;

-- ----------------------------------------------------------
-- Step 3
-- Remove old columns from the submodule UDM Table
-- ----------------------------------------------------------
-- Check and drop the legacy columns if it exists from the UDM_TABLE_NAME
-- Drop the column from any index that exists
-- Note Disabling this step because if this steps fails remaining script does not execute
-- BEGIN TRY
--   ALTER TABLE {UDM_TABLE_NAME}
--   DROP COLUMN IF EXISTS [Total_Prediction_Tran_Risk_Score];

--   ALTER TABLE {UDM_TABLE_NAME}
--   DROP COLUMN IF EXISTS [P2P_Single_UDM_Prediction];

--   ALTER TABLE {UDM_TABLE_NAME}
--   DROP COLUMN IF EXISTS [AutoML_Prediction];

--   ALTER TABLE {PredictionTableName}
--   DROP COLUMN IF EXISTS [Prediction];
-- END TRY
-- BEGIN CATCH
--   PRINT 'Can not delete columns'
-- END CATCH

-- ----------------------------------------------------------
-- Step 4
-- Remove old Display Names records from UDM Mapping Table
-- ----------------------------------------------------------
IF EXISTS(SELECT 1 FROM [App].[UDMMapping] WHERE DisplayName = 'AutoMLPrediction')
BEGIN
DELETE
FROM [App].[UDMMapping]
WHERE DisplayName = 'AutoMLPrediction';
END

IF EXISTS(SELECT 1 FROM [App].[UDMMapping] WHERE DisplayName = 'TotalMLRiskScore')
BEGIN
DELETE
FROM [App].[UDMMapping]
WHERE DisplayName = 'TotalMLRiskScore';
END
-- ----------------------------------------------------------
-- Step 5
-- Add new Display Names records to UDM Mapping Table
-- ----------------------------------------------------------
-- get last display order
DECLARE @LastDisplayOrder INT;

SELECT @LastDisplayOrder = MAX(DisplayOrder)
FROM [App].[UDMMapping]
WHERE ToTable = '{UDM_TABLE_NAME}'
	AND IsGridColumn = 1
	AND DisplayOrder IS NOT NULL;

DECLARE @Module NVARCHAR(MAX) = N'{Module}';
DECLARE @SubModule NVARCHAR(MAX) = N'{SubModule}';
DECLARE @DisplayNamePrediction NVARCHAR(MAX) = 'ML_Prediction';
DECLARE @DisplayNameRiskScore NVARCHAR(MAX) = 'ML_Risk_Score';

-- Add new Display Name for Prediction Column
IF NOT EXISTS(SELECT 1 FROM App.UDMMapping WHERE DisplayName = @DisplayNamePrediction AND ToTable = '{UDM_TABLE_NAME}' AND IsGridColumn = 1)
BEGIN
INSERT INTO [App].[UDMMapping] (
	[Module]
	, [SourceERPName]
	, [FromTable]
	, [ToColumn]
	, [ToTable]
	, [FromColumn]
	, [MappedColumns]
	, [IsGridColumn]
	, [IsMappingColumn]
	, [DisplayName]
	, [DisplayOrder]
	, [Critical]
	, [Important]
	, [Reporting]
	, [ToColumnToBeReplaced]
)
VALUES (
	@Module
	, NULL
	, NULL
	, '{PredictionColumnName}'
	, '{UDM_TABLE_NAME}'
	, '{PredictionColumnName}'
	, NULL
	, 1
	, 1
	, @DisplayNamePrediction
	, @LastDisplayOrder + 1
	, 0
	, 0
	, 1
	, 'i.[{PredictionColumnName}]'
);
END

-- Add new Display Name for ML Risk Score Column
IF NOT EXISTS(SELECT 1 FROM App.UDMMapping WHERE DisplayName = @DisplayNameRiskScore AND ToTable = '{UDM_TABLE_NAME}' AND IsGridColumn = 1)
BEGIN
INSERT INTO [App].[UDMMapping] (
	[Module]
	, [SourceERPName]
	, [FromTable]
	, [ToColumn]
	, [ToTable]
	, [FromColumn]
	, [MappedColumns]
	, [IsGridColumn]
	, [IsMappingColumn]
	, [DisplayName]
	, [DisplayOrder]
	, [Critical]
	, [Important]
	, [Reporting]
	, [ToColumnToBeReplaced]
)
VALUES (
	@Module
	, NULL
	, NULL
	, '{ML_Risk_Score_Column}'
	, '{UDM_TABLE_NAME}'
	, NULL
	, NULL
	, 1
	, 0
	, @DisplayNameRiskScore
	, @LastDisplayOrder + 2
	, 0
	, 0
	, 0
	, 'i.[{ML_Risk_Score_Column}]'
);
END
