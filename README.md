# Round 1A
## Our approach
- Using my PyMuPDF to parse the PDF
- Generating a list of features from the PDF for each line (such as, font size, x,y coordinate, space above, space below, is bold, etc.)
- Training an XGBoost model based on these features on a few manually labelled PDFs
## Models or Libraries used
- pymupdf
- scikit-learn
- pandas
- numpy
- xgboost-cpu (model)
## How to run
- Clone the repo using ` git clone ` and navigate into it.
- Build the docker image using:
	` docker build --platform linux/amd64 -t solution1a .`
- Run the image using:
	` docker run --rm -v /path/to/input/folder:/app/input -v /path/to/output/folder:/app/output --network none solution1a`
	