from orchestrator import Orchestrator

class Main:
    """
    Main class for executing the orchestrator.
    """

    def __init__(self):
        """
        Initializes the Main class.
        """
        self.orchestrator = Orchestrator()

    def execute(self):
        """
        Executes the Orchestrator instance.
        """
        self.orchestrator.run()


if __name__ == '__main__':
    main = Main()
    main.execute()