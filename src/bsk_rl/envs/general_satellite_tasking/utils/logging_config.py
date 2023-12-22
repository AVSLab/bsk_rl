import logging

sim_format = logging.Formatter(
    "%(asctime)s %(shortname)-30s %(levelname)-10s <%(sim_time)-.2f> %(message)s",
    defaults={"sim_time": -1},
)


class ContextFilter(logging.Filter):
    def __init__(self, name: str = "", env=None, proc_id=None) -> None:
        super().__init__(name)
        self.env = env
        self.proc_id = proc_id

    def filter(self, record):
        record.shortname = ".".join(record.name.split(".")[3:])
        try:
            record.sim_time = self.env.simulator.sim_time
        except AttributeError:
            pass
        return self.proc_id == record.process
