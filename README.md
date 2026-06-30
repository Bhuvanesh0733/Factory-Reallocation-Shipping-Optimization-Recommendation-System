# Factory Reallocation & Shipping Optimization System

This is a Streamlit dashboard built for Nassau Candy Distributor, or at least a version of it inspired by their shipping data. The basic question driving the whole thing is simple enough to state in one line: if a product currently ships out of one factory, would it get to customers faster (or more profitably) out of a different one? Answering that for real, at scale, across thousands of orders, is where it stops being simple.

**Live demo:** [factory-reallocation-shipping-optimization-recommendation-sys.com](https://factory-reallocation-shipping-optimization-recommendation-syst.streamlit.app/)

## What this actually does

The dataset is order-level shipping records: dates, regions, divisions, product names, costs, profits, that sort of thing. None of it mentions which factory makes which product, so the first real step is mapping each Wonka-branded item to one of five fictional factories (Lot's O' Nuts, Wicked Choccy's, Sugar Shack, Secret Factory, and The Other Factory) using a manually built lookup table. Once that's in place, the app calculates a delay metric, ship date minus order date, and tries to model it.

Three regression models get trained on the same feature set: a plain linear regression as a baseline, a random forest, and a gradient boosting regressor. Whichever one scores best on R² becomes "the model," and everything downstream, the simulator, the recommendation engine, leans on that winner. I'll be honest that this part felt a little uncomfortable to commit to. R² on shipping delay data is noisy almost by definition; a model that explains twelve percent of the variance isn't exactly something to brag about, but it's also not nothing, and for a recommendation system the relative comparison between factories often matters more than the absolute accuracy of any single prediction.

From there, the app simulates what would happen if every order had been routed through a different factory instead. That means re-running the trained model on a modified version of each row, swapping out the factory and its associated average delay, and checking whether the predicted delay actually improves. If it does, that combination becomes a recommendation, with the projected lead time reduction, a rough profit impact figure, and a risk reduction percentage attached to it.

There's also a clustering step tucked into the Analytics tab. Routes (a combination of division and region) get grouped by sales, units, profit, and average delay using k-means, mostly to surface which clusters of routes tend to run consistently slow. It's a smaller piece of the project, but it adds a layer that pure regression wouldn't catch on its own, namely whether certain region-product combinations are structurally troublesome regardless of which factory handles them.

## The dashboard itself

Five tabs, more or less in the order you'd want to read them:

- **Dashboard** gives you the headline numbers and two side-by-side charts of average delay by factory and the overall delay distribution.
- **Analytics** is where the scatter plots, the region-factory heatmap, the cluster visualization, and a "how often does each factory get recommended" chart all live.
- **Simulator** lets you pick a product and see, factory by factory, what the model predicts would happen to its delay, plus a little summary card up top and a map showing where each factory actually sits.
- **Recommendations** surfaces the best reassignment for whatever product and region you've filtered to in the sidebar, with a slider that lets you weight the ranking toward speed or toward profit.
- **Model Evaluation** is the straightforward one: MAE, RMSE, and R² for all three models side by side, plus feature importances if a tree-based model won.

A sidebar with product, region, and ship mode filters sits on the left throughout, along with that speed-versus-profit slider.

## Running it

You'll need Python with the packages in `requirements.txt`, installed via pip. After that:

```
streamlit run app.py
```

The CSV (`Nassau Candy Distributor.csv`) needs to sit in the same folder as `app.py`. The app won't go looking for it elsewhere, partly on purpose, since hardcoding a path that only exists on one person's laptop is exactly the kind of thing that breaks a project the moment someone else tries to run it.

## Things worth knowing before you trust the numbers

I want to flag a few rough edges rather than pretend this is a finished, audited piece of analytics software.

First, the delay figures are sensitive to how clean the Order Date and Ship Date columns actually are. If those two columns disagree wildly for even a modest fraction of rows, average delays can balloon into numbers that don't make intuitive sense, and the IQR-based outlier filter won't necessarily catch it, because if enough rows are affected, the quartiles themselves shift to accommodate the bad data. So if the dashboard ever shows something like an 800-day average delay, that's almost certainly a data quality issue upstream rather than a genuine finding about shipping logistics, and it's worth tracing back to the source file before reading too much into it.

Second, the "best model" is chosen by R² alone. That's a defensible choice, but it's not the only one. A model with a slightly lower R² but a meaningfully lower MAE might be the more useful pick depending on what you're optimizing for, and the app doesn't currently give you a way to override that selection manually.

Third, the profit impact figures are a simplification. They're calculated as lead time reduction multiplied by the order's existing gross profit, which is a reasonable first approximation but doesn't account for things like the actual cost of physically reallocating production, contractual obligations with existing factories, or capacity constraints at the "better" factory. In other words, the model might recommend moving everything to one factory because it has the best average numbers, without knowing or caring whether that factory could actually handle the volume.

None of this means the tool is useless. It means the recommendations are a starting point for a conversation, not a final verdict.

## Tech stack

Streamlit for the interface, scikit-learn for the modeling and clustering, pandas and numpy for the data wrangling, and Plotly for the charts and the map.

## Project structure

```
.
├── app.py
├── requirements.txt
├── diagnose_dates.py
└── Nassau Candy Distributor.csv
```

`diagnose_dates.py` is a small standalone script for checking whether the date columns in the source CSV are parsing the way they should, separate from the main app. Worth running once before trusting the delay numbers, especially on a new export of the data.
