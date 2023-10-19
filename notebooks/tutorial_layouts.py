from typing import Optional, Union, Dict, List, Any
from renumics.spotlight import layout
from renumics.spotlight.layout import (
    Layout,
    Tab,
    Split,
    lenses,
    table,
    similaritymap,
    inspector,
    split,
    tab,
    metric,
    issues,
    confusion_matrix,
    histogram,
    scatterplot
)
from renumics.spotlight.dtypes import create_dtype, is_audio_dtype, is_image_dtype


def debug_image_classification(
    image: str='image',
    label: str = "label",
    prediction: str = "prediction",
    emb_x: str = "emb_x",
    emb_y: str = "emb_y",
) -> Layout:
    """This function generates a Spotlight layout for debugging a machine learning classification model.

    Args:
        label: Name of the column that contains the label.
        prediction: Name of the column that contains the prediction.
        embedding: Name of the column that contains the embedding.
        inspect: Name and type of the columns that are displayed in the inspector, e.g. {'audio': spotlight.dtypes.audio_dtype}.
        features: Names of the columns that contain useful metadata and features.

    Returns:
        The configured layout for `spotlight.show`.
    """

    # first column: table + issues
    metrics = tab(
        metric(name="Accuracy", metric="accuracy", columns=[label, prediction]),
        weight=15,
    )
    column1 = split(
        [metrics, tab(table(), weight=65)], weight=80, orientation="horizontal"
    )
    column1 = split(
        [column1, tab(issues(), weight=40)], weight=80, orientation="horizontal"
    )

    column2_list = []
    column2_list.append(
        tab(
            confusion_matrix(
                name="Confusion matrix", x_column=label, y_column=prediction
            ),
            weight=40,
        )
    )
   
    row3 = tab(
        scatterplot(name="Embedding", x_column=emb_x, y_column=emb_y, color_by_column=label),
        weight=40,
    )
    column2_list.append(row3)

    column2: Union[Tab, Split]

    column2 = split(column2_list, orientation="horizontal")
    

    # fourth column: inspector
    inspector_fields = []
   
    
       
    inspector_fields.append(lenses.image('image'))
   

    inspector_fields.append(lenses.scalar(label))
    inspector_fields.append(lenses.scalar(prediction))

    inspector_view = inspector("Inspector", lenses=inspector_fields, num_columns=4)


    # build everything together
    column2.weight = 40
    half1 = split([column1, column2], weight=80, orientation="vertical")
    half2 = tab(inspector_view, weight=40)

    nodes = [half1, half2]

    the_layout = layout.layout(nodes)

    return the_layout




def compare_image_classification(
    label: str = "label",
    model1_prediction: str = "prediction",
    model1_emb_x: str = "emb_x",
    model1_emb_y: str = "emb_y",
    model1_correct: str = "model1_correct",
    model2_prediction: str = "model2_prediction",
    model2_emb_x: str = "mmodel2_emb_x",
    model2_emb_y: str = "model2_emb_y",
    model2_correct: str = "model2_correct",
) -> Layout:
    """This function generates a Spotlight layout for comparing two different machine learning classification models.

    Args:
        label: Name of the column that contains the label.
        model1_prediction: Name of the column that contains the prediction for model 1.
        model1_embedding: Name of the column that contains thee embedding for model 1.
        model1_correct: Name of the column that contains a flag if the data sample is predicted correctly by model 1.
        model2_prediction: Name of the column that contains the prediction for model 2.
        model2_embedding: Name of the column that contains thee embedding for model 2.
        model2_correct: Name of the column that contains a flag if the data sample is predicted correctly by model 2.
        inspect: Name and type of the columns that are displayed in the inspector, e.g. {'audio': spotlight.dtypes.audio_dtype}.

    Returns:
        The configured layout for `spotlight.show`.
    """

    # first column: table + issues
    metrics = split(
        [
            tab(
                metric(
                    name="Accuracy model 1",
                    metric="accuracy",
                    columns=[label, model1_prediction],
                )
            ),
            tab(
                metric(
                    name="Accuracy model 2",
                    metric="accuracy",
                    columns=[label, model2_prediction],
                )
            ),
        ],
        orientation="vertical",
        weight=15,
    )
    column1 = split(
        [metrics, tab(table(), weight=65)], weight=80, orientation="horizontal"
    )
    column1 = split(
        [column1, tab(issues(), weight=40)], weight=80, orientation="horizontal"
    )

    column2_list = []
    column2_list.append(
        tab(
            confusion_matrix(
                name="Model 1 confusion matrix",
                x_column=label,
                y_column=model1_prediction,
            ),
            confusion_matrix(
                name="Model 2 confusion matrix",
                x_column=label,
                y_column=model2_prediction,
            ),
            weight=40,
        )
    )

    # third column: similarity maps
   
    row2 = tab(
        confusion_matrix(
            name="Model1 vs. Model2 - binned scatterplot",
            x_column=model1_correct,
            y_column=model2_correct,
        ),
        weight=40,
    )
    column2_list.append(row2)

    
    row3 = tab(
        scatterplot(name="Model 1 embedding", x_column=model1_emb_x, y_column=model1_emb_y, color_by_column=label),
        scatterplot(name="Model 2 embedding", x_column=model2_emb_x, y_column=model2_emb_y, color_by_column=label),
        weight=40,
    )

    column2_list.append(row3)

    column2: Union[Tab, Split]

    column2 = split(column2_list, orientation="horizontal")
  

    # fourth column: inspector
    inspector_fields = []
   
    
       
    inspector_fields.append(lenses.image('image'))
   

    inspector_fields.append(lenses.scalar(label))
    inspector_fields.append(lenses.scalar(model1_prediction))
    inspector_fields.append(lenses.scalar(model2_prediction))

    inspector_view = inspector("Inspector", lenses=inspector_fields, num_columns=4)

    # build everything together
    column2.weight = 40
    half1 = split([column1, column2], weight=80, orientation="vertical")
    half2 = tab(inspector_view, weight=40)

    nodes = [half1, half2]

    the_layout = layout.layout(nodes)

    return the_layout