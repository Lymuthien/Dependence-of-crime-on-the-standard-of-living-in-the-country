import pandas as pd

from src.dependence_analyzer import DependenceAnalyzer


class App(object):
    def __init__(self, directory: str):
        self._dir = directory
        self._analyzer = DependenceAnalyzer()

    def _read_df(self):
        gdp_df = pd.read_csv(f"{self._dir}gdp_per_capita.csv")
        quality_of_life_df = pd.read_csv(f"{self._dir}quality_of_life.csv")
        crime_df = pd.read_csv(f"{self._dir}crime_index.csv")
        murder_df = pd.read_csv(f"{self._dir}murder_percent.csv")

        return gdp_df, quality_of_life_df, crime_df, murder_df

    @staticmethod
    def _create_df():
        columns = [
            'COMPARER_COLUMN', 'AVERAGE_COMPARER', 'AVERAGE_COMPARABLE_HIGHER', 'AVERAGE_COMPARABLE_LOWER',
            'DIFFERENCE', 'DIFFERENCE_PERCENT', 'CORRELATION', 'P_VALUE'
        ]
        return pd.DataFrame(columns=columns)

    def run(self):
        try:
            gdp_df, quality_of_life_df, crime_df, murder_df = self._read_df()
        except Exception as e:
            print(f"Error reading dataframes: {e}")
            return

        gdp_and_crime_df = pd.merge(gdp_df, crime_df, on="COUNTRY")
        crime_and_qol_df = pd.merge(crime_df, quality_of_life_df, on="COUNTRY")
        gdp_and_murder_df = pd.merge(gdp_df, murder_df, on="COUNTRY")
        murder_and_qol_df = pd.merge(murder_df, quality_of_life_df, on="COUNTRY")

        crime_df = self._create_df()
        murder_df = self._create_df()

        self._analyzer.calculate_dependence_by_column(
            gdp_and_crime_df, crime_df, "GDP", "CRIME_INDEX"
        )
        for comparer_column in self._analyzer.COMPARER_COLUMNS:
            self._analyzer.calculate_dependence_by_column(
                crime_and_qol_df, crime_df, comparer_column, "CRIME_INDEX"
            )

        crime_df.to_csv(f"{self._dir}crime_dependencies.csv")

        self._analyzer.calculate_dependence_by_column(
            gdp_and_murder_df, murder_df, "GDP", "RATE_PER_100000_N"
        )
        for comparer_column in self._analyzer.COMPARER_COLUMNS:
            self._analyzer.calculate_dependence_by_column(
                murder_and_qol_df, murder_df, comparer_column, "RATE_PER_100000_N"
            )

        murder_df.to_csv(f"{self._dir}murder_dependencies.csv")


if __name__ == "__main__":
    app = App("data/")
    app.run()
