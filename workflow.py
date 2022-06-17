import mlflow as mf


def main():
    with mf.start_run() as active_run:
        print("Launching 'download'")
        step1 = mf.run(".", "preprocess", parameters={})
        # step1 = mf.tracking.MlflowClient().get_run(step1.run_id)

        step2 = mf.run(".", "train", parameters={})


if __name__ == "__main__":
    main()
