install:
	pip install --upgrade pip &&\
		pip install -r ./model_development/requirements.txt

format:	
	black .

train:
	python ./model_development/train_model.py	

eval:
	@threshold=0.95; \
	f1_score=$$(grep "F1 Score" ./model_reporting/metrics.txt | awk -F": " '{print $$2}'); \
	echo "F1 Score: $$f1_score"; \
	if [ $$(echo "$$f1_score < $$threshold" | bc) -eq 1 ]; then \
		echo "Build failed: F1 Score ($$f1_score) is below the threshold ($$threshold)."; \
		exit 1; \
	else \
		echo "F1 Score ($$f1_score) meets the threshold ($$threshold)."; \
	fi

report:
	echo "## Model Metrics" > report.md
	cat ./model_reporting/metrics.txt >> report.md
	
	echo '\n## Model Confusion Matrix' >> report.md
	echo '![Confusion Matrix](./model_reporting/confusion_matrix.png)' >> report.md

	echo '\n## Model Training Accuracy Curve' >> report.md
	echo '![Model Training Accuracy Curve](./model_reporting/nn_acc_curve.png)' >> report.md

	echo '\n## Model Training Loss Curve' >> report.md
	echo '![Model Training Loss Curve](./model_reporting/nn_loss_curve.png.png)' >> report.md

	echo '\n## Model Training Learning Rate Curve' >> report.md
	echo '![Model Training Learning Rate Curve](./model_reporting/nn_lr_curve.png.png)' >> report.md
	
	cml comment create report.md

load-model:
	cp ./model_development/model_architecture.py ./app/model/
	echo 'Model Architecture Loaded'

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	@if ! git diff-index --quiet HEAD --; then \
		git commit -am "Update with new results"; \
		git push --force origin HEAD:update; \
	else \
		echo "No changes to commit."; \
 	fi

all: install format train eval report load-model update-branch

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git push --force origin update
	git pull --rebase origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential


	# pip install -U "huggingface_hub[cli]"
	# git config --global user.name $(USER_NAME)
	# git config --global user.email $(USER_EMAIL)
	# git pull --allow-unrelated-histories --no-rebase origin update
	# git switch update
	# huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload MarkJamesDunbar/mnist-interactive-app ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload MarkJamesDunbar/mnist-interactive-app ./model_reporting /model_reporting --repo-type=space --commit-message="Sync Model Report"

deploy: hf-login push-hub


