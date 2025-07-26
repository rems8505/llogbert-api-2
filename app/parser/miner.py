class DummyMiner:
    def __init__(self):
        self.template_map = {}

    def add_log_message(self, line):
        # Dummy: map each line to a fake cluster_id (hash based)
        cluster_id = abs(hash(line)) % 100
        return {"cluster_id": cluster_id}
