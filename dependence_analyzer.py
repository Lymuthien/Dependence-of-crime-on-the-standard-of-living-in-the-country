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
        :type df: pd.DataFrame
        :param comparer_column: Column in the DataFrame used to create groups based on its median.
        :type comparer_column: str
        :param comparable_column: Column to analyze in relation to the comparer column.
        :type comparable_column: str
        :return: A tuple containing the median of the comparer column, the median of the comparable column,
            the median of the comparable column for the high comparer group, the median of the comparable column
            for the low comparer group, the correlation coefficient between the two columns, and the p-value from
            the two-sample t-test.
        :rtype: Tuple[float, float, float, float, float, float]
        """

        median_comparer = df[comparer_column].median()
        median_comparable = df[comparable_column].median()

        group_high_comparer = df[df[comparer_column] > median_comparer]
        group_low_comparer = df[df[comparer_column] <= median_comparer]

        median_comparable_high = group_high_comparer[comparable_column].median()
        median_comparable_low = group_low_comparer[comparable_column].median()

        correlation = df[comparer_column].corr(df[comparable_column])

        group_high = group_high_comparer[comparable_column].dropna()
        group_low = group_low_comparer[comparable_column].dropna()
        t_stat, p_value = ttest_ind(group_high, group_low)

        return median_comparer, median_comparable, median_comparable_high, median_comparable_low, correlation, p_value.round(3)

    @staticmethod
    def _format_column_name(
            column_name: str,
    ) -> str:
        """
        Formats a given column name by replacing underscores with spaces and converting
        the string to lowercase.

        :param column_name: The name of the column to format.
        :type column_name: str
        :return: The formatted column name as a lowercase string with underscores replaced by spaces.
        :rtype: str
        """
        return column_name.replace('_', ' ').lower()

    def _print_dependence_statistics(
            self,
            comparer_column: str,
            comparable_column: str,
            avg_comparer: float,
            avg_comparable_high: float,
            avg_comparable_low: float,
            avg_comparable: float,
            corr: float,
            p_value: float,
    ) -> str:
        """
        Creates statistical information about the dependence between two variables, providing insights
        such as averages, differences, correlation, and statistical significance.

        :param comparer_column: Name of the column being used as the baseline for comparison.
        :param comparable_column: Name of the column being compared against the baseline.
        :param avg_comparer: Average value of the comparer column.
        :param avg_comparable_high: Average value of the comparable column for entries with above-average
            values in the comparer column.
        :param avg_comparable_low: Average value of the comparable column for entries with below-average
            values in the comparer column.
        :param avg_comparable: Overall average value of the comparable column across all entries.
        :param corr: Correlation coefficient between the comparer and comparable columns.
        :param p_value: P-value indicating the statistical significance of the difference in the comparable
            column's values.
        :return: String representation of the statistical information.
        """

        diff = abs(avg_comparable_high - avg_comparable_low)
        formatted_comparer = self._format_column_name(comparer_column)
        formatted_comparable = self._format_column_name(comparable_column)

        return (f"\nAverage {formatted_comparer}: {avg_comparer},\n"
                f"Average {formatted_comparable} for countries with above-average {formatted_comparer}: {avg_comparable_high}.\n"
                f"Average {formatted_comparable} for countries with below-average {formatted_comparer}: {avg_comparable_low}.\n"
                f"Difference: {diff}\n"
                f"Percent from average {formatted_comparable} value: {diff / avg_comparable * 100:.2f}%\n"
                f"Correlation: {corr:.2f}\n"
                f"P-value for difference in {formatted_comparable}: {p_value:.4f}\n")

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
        Calculate and analyze the dependence between two columns in a DataFrame.

        This method computes statistical parameters to examine the dependence between a comparer
        column and a comparable column in the provided DataFrame. It involves statistical calculations,
        displays the calculated statistics, and creates visualizations to represent the relationship between
        the columns.

        :param df: DataFrame containing the data to perform dependence analysis on.
        :param df_result: DataFrame to store the calculated statistics.
        :param comparer_column: Column used as the reference for comparison.
        :param comparable_column: Column whose dependence on the comparer_column is analyzed.
        """

        avg_comparer, avg_comparable, avg_comparable_high, avg_comparable_low, corr, p_value = self.calculate_dependence_statistics(
            df, comparer_column, comparable_column
        )

        self._save_stats_to_df(
            df_result, comparer_column, avg_comparer, avg_comparable_high, avg_comparable_low,
            avg_comparable, corr, p_value)

        #
        # self._print_dependence_statistics(
        #     comparer_column, comparable_column, avg_comparer, avg_comparable_high, avg_comparable_low, avg_comparable,
        #     corr, p_value
        # )

        # self._plot_dependence(df, comparer_column, comparable_column)


class App(object):
    def __init__(self):
        self._analyzer = DependenceAnalyzer()

    @staticmethod
    def _read_df():
        gdp_df = pd.read_csv("gdp_per_capita.csv")
        quality_of_life_df = pd.read_csv("quality_of_life.csv")
        crime_df = pd.read_csv("crime_index.csv")
        murder_df = pd.read_csv("murder_percent.csv")

        return gdp_df, quality_of_life_df, crime_df, murder_df

    def run(self):
        try:
            gdp_df, quality_of_life_df, crime_df, murder_df = self._read_df()
        except Exception as e:
            print(f"Error reading dataframes: {e}")
            return

        gdp_and_crime_df = pd.merge(gdp_df, crime_df, on="COUNTRY")
        crime_and_qol_df = pd.merge(crime_df, quality_of_life_df, on="COUNTRY")

        crime_df = pd.DataFrame(columns=[
            'COMPARER_COLUMN', 'AVERAGE_COMPARER', 'AVERAGE_COMPARABLE_HIGHER', 'AVERAGE_COMPARABLE_LOWER',
            'DIFFERENCE', 'DIFFERENCE_PERCENT', 'CORRELATION', 'P_VALUE'
        ])
        self._analyzer.calculate_dependence_by_column(gdp_and_crime_df, crime_df, "GDP", "CRIME_INDEX", )
        for comparer_column in self._analyzer.COMPARER_COLUMNS:
            self._analyzer.calculate_dependence_by_column(crime_and_qol_df, crime_df, comparer_column, "CRIME_INDEX")

        crime_df.to_csv("crime_dependencies.csv")


        #
        # gdp_and_murder_df = pd.merge(gdp_df, murder_df, on="COUNTRY")
        # murder_and_qol_df = pd.merge(murder_df, quality_of_life_df, on="COUNTRY")
        #
        # self._analyzer.calculate_dependence_by_column(gdp_and_murder_df, "GDP", "RATE_PER_100000_N")
        # for comparer_column in self._analyzer.COMPARER_COLUMNS:
        #     self._analyzer.calculate_dependence_by_column(murder_and_qol_df, comparer_column, "RATE_PER_100000_N")
        #

if __name__ == "__main__":
    app = App()
    app.run()
