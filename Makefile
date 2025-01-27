format:	
	black *.py 

train:
	python ./model_development/train_model.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md
	
	cml comment create report.md

load-model:
	## Load the model and the model architecture into the app/model folder
		
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	@if ! git diff-index --quiet HEAD --; then \
		git commit -am "Update with new results"; \
		git push --force origin HEAD:update; \
	else \
		echo "No changes to commit."; \
	fi

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull --no-rebase origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload MarkJamesDunbar/Drug-Classification ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload MarkJamesDunbar/Drug-Classification ./model /model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload MarkJamesDunbar/Drug-Classification ./results /metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train eval update-branch deploy