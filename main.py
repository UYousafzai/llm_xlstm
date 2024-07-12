from experiments import GermanWikiTestExperiment


if __name__ == "__main__":
    experiment = GermanWikiTestExperiment()
    experiment.warmup()
    experiment.run()