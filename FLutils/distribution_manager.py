from multiprocessing.managers import BaseManager

class DictItem:
    def __init__(self, clients):
        self.items = {"DirName": ""}
        self.clients = clients
        self.creat_client()

    def creat_client(self):
        for ind in range(self.clients):
            self.set(key = f"client_{ind}", value = None)
    def set(self, key, value):
        self.items[key] = value
    def get(self, key):
        return self.items.get(key)
    def __setitem__(self, key, value):
        self.set(key, value)

class ManagerServer:
    def __init__(self, domain, port, auth_key, clients):
        self.auth_key = auth_key
        self.is_stop = 0
        self.dict = {}
        self.open_lock = None
        d = DictItem(clients)
        BaseManager.register('dict', callable=lambda: d)
        self.server_manager = BaseManager(address=(domain, port), authkey=self.auth_key)

    def get_dict(self):
        self.dict = self.server_manager.dict()
        return self.dict

    def run(self):
        self.server_manager.start()

    def stop(self):
        self.server_manager.shutdown()
        self.is_stop = 1

class ManagerClient:
    def __init__(self, domain, port, auth_key):
        self.dict = {}
        self.open_lock = None
        BaseManager.register('dict')
        BaseManager.register('open_lock')
        self.client_manager = BaseManager(address=(domain, port), authkey=auth_key)
        self.client_manager.connect()

    def get_dict(self):
        self.dict = self.client_manager.dict()
        return self.dict

if __name__ == '__main__':
    pass