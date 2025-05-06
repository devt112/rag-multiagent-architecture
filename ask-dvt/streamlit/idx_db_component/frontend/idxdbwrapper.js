
async function initDb(db, store, version, key, idx, uniq=false) {
    return new Promise((resolve, reject) => {        
        let request = indexedDB.open(db, version);

        request.onerror = event => {
            alert('Error Event, check console');
            console.error(event);
        }
        
        request.onupgradeneeded = event => {
            console.log('idb onupgradeneeded firing');
            let db = event.target.result;
            let objectStore = db.createObjectStore(store, { keyPath: key, autoIncrement:true });
            objectStore.createIndex(idx, idx, { unique: uniq });
        };
        
        request.onsuccess = event => {
            resolve(event.target.result);
        };
    });
}

async function getAllRecords(db, s) {
    return new Promise((resolve, reject) => {
        let transaction = db.transaction([s], 'readonly');
        
        transaction.onerror = event => {
            reject(event);
        };
        
        let store = transaction.objectStore(s);
        store.getAll().onsuccess = event => {
            resolve(event.target.result);
        };    
    });
}

async function getAllRecordsByKey(db, s, k, idxname) {
    return new Promise((resolve, reject) => {
        let transaction = db.transaction([s], 'readonly');
        var objectStore = transaction.objectStore("logs");
        const index = objectStore.index(idxname);
        let query = index.getAll(k);

        query.onsuccess = function(event) { resolve(event.target.result); };
        query.onerror = function(event) { reject(event); };  
        // transaction.oncomplete  = function(event) { db.close(); };  
    });
}

async function createRecord(db, s, data) {
    return new Promise((resolve, reject) => {        
        let transaction = db.transaction([s], 'readwrite');
        transaction.oncomplete = event => {
            resolve();
        };
        
        transaction.onerror = event => {
            reject(event);
        };
        
        let store = transaction.objectStore(s);
        store.put(data);
    });
}

async function removeRecord(db, store, key) {
    return new Promise((resolve, reject) => {
        let transaction = db.transaction([store], 'readwrite');
        transaction.oncomplete = event => {
            resolve();
        };
        
        transaction.onerror = event => {
            reject(event);
        };
        
        let store = transaction.objectStore(store);
        store.delete(key);        
    });
}

function deleteDB(db) {
    let DBDeleteRequest = window.indexedDB.deleteDatabase(db);

    DBDeleteRequest.onerror = (event) => {
    console.error("Error deleting database.");
    };

    DBDeleteRequest.onsuccess = (event) => {
    console.log("Database deleted successfully");

    console.log(event.result); // should be undefined
    };
}