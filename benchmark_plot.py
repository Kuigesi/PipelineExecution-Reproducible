import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


if __name__=="__main__":

    # Load data
    df = pd.read_csv('./data/benchmark.csv', index_col=0)

    # rename index
    df.rename(index={'Data Parallism (2 GPU)': 'Data Parallism\n(2 GPU)',\
       'Data Parallism (4 GPU)': 'Data Parallism\n(4 GPU)', 'Mixed Parallism (no pipeline)':\
       'Mixed Parallism\n(no pipeline)', 'Mixed Parallism (4 pipelines)': 'Mixed Parallism\n(4 pipelines)',\
       'Mixed Parallism (8 pipelines)': 'Mixed Parallism\n(8 pipelines)'}, inplace=True)
    df['Runtime'] = df['Runtime'].round(3)

    # plot runtime
    ax = df.plot.bar(grid=False, figsize=(8,4))
    plt.xticks(rotation=0)
    (low, high) = plt.ylim()
    plt.ylim(low, high*1.1)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("Runtime of 40 iterations (sec)")
    for col in df.columns:
      for id, val in enumerate(df[col]):
        ax.text(id, val+1.4, str(val), va="top", ha="center")
    plt.legend().remove()
    plt.savefig("./pictures/pipelineparallelruntime.pdf", format="pdf", bbox_inches="tight")

    # plot speed up
    dfspeedup = df.rename(columns={'Runtime':'Speed Up'})
    baseline = dfspeedup.loc['Data Parallism\n(2 GPU)']['Speed Up']
    dfspeedup['Speed Up'] = baseline / dfspeedup['Speed Up']
    dfspeedup['Speed Up'] = dfspeedup['Speed Up'].round(2)
    axspeed = dfspeedup.plot.bar(grid=False, figsize=(8,4))
    axspeed.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.xticks(rotation=0)
    (low, high) = plt.ylim()
    plt.ylim(low, high*1.1)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("Speed Up")
    for col in dfspeedup.columns:
      for id, val in enumerate(dfspeedup[col]):
        axspeed.text(id, val+0.14, str(round(val*100))+'%', va="top", ha="center")
    plt.legend().remove()
    plt.savefig("./pictures/pipelineparallelspeedup.pdf", format="pdf", bbox_inches="tight")
