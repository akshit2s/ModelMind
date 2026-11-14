import pandas as pd


def create_tool_registry(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    # Tool 1: Inspect the complete dataset
    def inspect_dataset() -> dict:
        """
        Returns a structural summary of the currently loaded dataset.

        Includes row count, column count, column names,
        data types, missing values and duplicate rows.

        Does not return actual dataset records.
        """

        columns = []

        for column in df.columns:

            columns.append({
                "name": str(column),
                "data_type": str(df[column].dtype),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique())
            })

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_information": columns,
            "duplicate_rows": int(df.duplicated().sum())
        }

    # Tool 2: Inspect an individual column
    def inspect_column(column_name: str) -> dict:
        """
        Returns structural information about a specified column.

        Args:
            column_name: Name of the column to inspect.

        Does not return the actual values stored in the column.
        """

        if column_name not in df.columns:
            return {
                "error": f"Column '{column_name}' does not exist."
            }

        column = df[column_name]

        if pd.api.types.is_numeric_dtype(column):
            column_type = "numerical"

        elif pd.api.types.is_bool_dtype(column):
            column_type = "boolean"

        else:
            column_type = "categorical"

        return {
            "column": column_name,
            "data_type": str(column.dtype),
            "column_type": column_type,
            "missing_values": int(column.isna().sum()),
            "unique_values": int(column.nunique()),
            "total_records": int(len(column))
        }

    # Register available tools
    return {
        "inspect_dataset": inspect_dataset,
        "inspect_column": inspect_column
    }