venv:
	source venv311/bin/activate

wordcloud:
	source freshenv/bin/activate && python plot_wordcloud.py
