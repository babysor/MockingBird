from typing import List

from pydantic import BaseModel


class ScoredLabel(BaseModel):
    label: str
    score: float


class ClassificationOutput(BaseModel):
    __root__: List[ScoredLabel]

    def __iter__(self):  # type: ignore
        return iter(self.__root__)

    def __getitem__(self, item):  # type: ignore
        return self.__root__[item]

    def render_output_ui(self, streamlit) -> None:  # type: ignore
        import plotly.express as px

        sorted_predictions = sorted(
            [prediction.dict() for prediction in self.__root__],
            key=lambda k: k["score"],
        )

        num_labels = len(sorted_predictions)
        if len(sorted_predictions) > 10:
            num_labels = streamlit.slider(
                "Maximum labels to show: ",
                min_value=1,
                max_value=len(sorted_predictions),
                value=len(sorted_predictions),
            )
        fig = px.bar(
            sorted_predictions[len(sorted_predictions) - num_labels :],
            x="score",
            y="label",
            orientation="h",
        )
        streamlit.plotly_chart(fig, use_container_width=True)
        # fig.show()
