import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns


class DependenceAnalyzer(object):
    COMPARER_COLUMNS = (
        "QUALITY_OF_LIFE_INDEX", "PURCHASING_POWER_INDEX", "HEALTH_CARE_INDEX",
        "COST_OF_LIVING_INDEX", "PROPERTY_PRICE_TO_INCOME_RATIO", "POLLUTION_INDEX", "CLIMATE_INDEX"
    )

    @staticmethod
    def calculate_dependence_statistics(
            df: pd.DataFrame,
            comparer_column: str,
            comparable_column: str,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Calculates statistical dependences and relationships between two specified columns in a given DataFrame.

        This method computes medians for the comparer and comparable columns, divides the data into two groups
        based on the median of the comparer column, and determines median values for the comparable column
        within these groups. Additionally, it calculates the correlation between the columns and performs a
        two-sample t-test to assess statistical differences between the high and low comparer groups.

        :param df: DataFrame containing the columns to analyze.
        :param comparer_column: Column in the DataFrame used to create groups based on its median.
        :param comparable_column: Column to analyze in relation to the comparer column.
        :return: A tuple containing the median of the comparer column, the median of the comparable column,
            the median of the comparable column for the high comparer group, the median of the comparable column
            for the low comparer group, the correlation coefficient between the two columns, and the p-value from
            the two-sample t-test.
        """

        median_comparer = df[comparer_column].median().round(3)
        median_comparable = df[comparable_column].median().round(3)

        group_high_comparer = df[df[comparer_column] > median_comparer].round(3)
        group_low_comparer = df[df[comparer_column] <= median_comparer].round(3)

        median_comparable_high = group_high_comparer[comparable_column].median().round(3)
        median_comparable_low = group_low_comparer[comparable_column].median().round(3)

        correlation = df[comparer_column].corr(df[comparable_column]).round(3)

        group_high = group_high_comparer[comparable_column].dropna()
        group_low = group_low_comparer[comparable_column].dropna()
        t_stat, p_value = ttest_ind(group_high, group_low)
        p_value = p_value.round(3)

        return median_comparer, median_comparable, median_comparable_high, median_comparable_low, correlation, p_value

    @staticmethod
    def _save_stats_to_df(
            df: pd.DataFrame,
            comparer_column: str,
            avg_comparer: float,
            avg_comparable_high: float,
            avg_comparable_low: float,
            avg_comparable: float,
            corr: float,
            p_value: float,
    ):
        """
        Saves statistical information into a given DataFrame. This includes details
        such as average values, differences, correlation, and statistical p-value.

        :param df: The DataFrame where the statistics will be saved.
        :param comparer_column: The column name in the DataFrame that represents the comparer.
        :param avg_comparer: The average value of the comparer column.
        :param avg_comparable_high: The higher average value of the comparable column.
        :param avg_comparable_low: The lower average value of the comparable column.
        :param avg_comparable: The average of the comparable values.
        :param corr: The correlation coefficient between the comparer and comparable columns.
        :param p_value: The p-value representing the significance of the correlation.
        """

        diff = abs(avg_comparable_high - avg_comparable_low)

        df.loc[comparer_column] = {
            'COMPARER_COLUMN': comparer_column,
            'AVERAGE_COMPARER': avg_comparer,
            'AVERAGE_COMPARABLE_HIGHER': avg_comparable_high,
            'AVERAGE_COMPARABLE_LOWER': avg_comparable_low,
            'DIFFERENCE': diff,
            'DIFFERENCE_PERCENT': diff / avg_comparable * 100,
            'CORRELATION': corr,
            'P_VALUE': p_value
        }

    @staticmethod
    def _plot_dependence(
            df: pd.DataFrame,
            comparer_column: str,
            comparable_column: str
    ):
        """
        Generate a scatter plot with regression line to visualize the relationship
        between two specified columns of a given DataFrame.

        :param df: A pandas DataFrame containing the data to plot.
        :param comparer_column: The name of the column to use for the x-axis of the scatter plot.
        :param comparable_column: The name of the column to use for the y-axis of the scatter plot.
        """

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=comparer_column, y=comparable_column, data=df)
        sns.regplot(x=comparer_column, y=comparable_column, data=df, scatter=False, color='red')
        plt.title(f"{comparer_column} vs {comparable_column}")
        plt.show()

    def calculate_dependence_by_column(
            self,
            df: pd.DataFrame,
            df_result: pd.DataFrame,
            comparer_column: str,
            comparable_column: str,
    ) :
        """
        Calculate dependence statistics between columns in a dataframe and save the
        results into another dataframe.

        :param df: Input dataframe containing the data to analyze.
        :param df_result: Dataframe to store the computed statistics.
        :param comparer_column: Column name in the input dataframe to use as the
            comparer in the statistical analysis.
        :param comparable_column: Column name in the input dataframe used as the
            comparable against the comparer column in the analysis.
        """

        avg_comparer, avg_comparable, avg_comparable_high, avg_comparable_low, corr, p_value = self.calculate_dependence_statistics(
            df, comparer_column, comparable_column
        )

        self._save_stats_to_df(
            df_result, comparer_column, avg_comparer, avg_comparable_high, avg_comparable_low,
            avg_comparable, corr, p_value)
