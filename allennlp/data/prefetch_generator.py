import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn')
from queue import Empty

class ExceptionItem(object):
    def __init__(self, exception):
        self.exception = exception

class PrefetchGeneratorException(Exception):
    pass

class GeneratorDied(PrefetchGeneratorException):
    pass

class PrefetchGenerator(object):
    def __init__(self, orig_gen, max_lookahead=None, get_timeout=10):
        """
        Creates a prefetch generator from a normal one.
        The elements will be prefetched up to max_lookahead
        ahead of the consumer. If max_lookahead is None,
        everything will be fetched.
        The get_timeout parameter is the number of seconds
        after which we check that the subprocess is still
        alive, when waiting for an element to be generated.
        Any exception raised in the generator will
        be forwarded to this prefetch generator.
        """
        if max_lookahead:
            self.queue = multiprocessing.Queue(max_lookahead)
        else:
            self.queue = multiprocessing.Queue()

        def wrapped():
            try:
                for item in orig_gen:
                    self.queue.put(item)
                raise StopIteration()
            except Exception as exception:  # pylint: disable=broad-except
                self.queue.put(ExceptionItem(exception))

        self.get_timeout = get_timeout

        self.process = multiprocessing.Process(target=wrapped)
        self.process_started = False

    def __enter__(self):
        """
        Starts the process
        """
        self.process.start()
        self.process_started = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Kills the process
        """
        if self.process and self.process.is_alive():
            self.process.terminate()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        if not self.process_started:
            raise PrefetchGeneratorException(
                    "The generator has not been started. "
                    "Please use \"with PrefetchGenerator(..) as g:\"")
        try:
            item_received = False
            while not item_received:
                try:
                    item = self.queue.get(timeout=self.get_timeout)
                    item_received = True
                except Empty:
                    # check that the process is still alive
                    if not self.process.is_alive():
                        raise GeneratorDied("The generator died unexpectedly.")

            if isinstance(item, ExceptionItem):
                raise item.exception
            return item

        except Exception:
            self.queue = None
            if self.process.is_alive():
                self.process.terminate()
            self.process = None
            raise
