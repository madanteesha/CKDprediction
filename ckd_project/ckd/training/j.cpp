#include <stdio.h>
#include <stdlib.h>

int *hashTable;
int TABLE_SIZE;

void initHashTable() {
    hashTable = (int *)malloc(TABLE_SIZE * sizeof(int));
    for (int i = 0; i < TABLE_SIZE; i++)
        hashTable[i] = -1;
}

void insert(int key) {
    int index = key % TABLE_SIZE;
    int original = index;

    while (hashTable[index] != -1) {
        index = (index + 1) % TABLE_SIZE;
        if (index == original) {
            printf("Hash Table is full. Cannot insert %d\n", key);
            return;
        }
    }
    hashTable[index] = key;
    printf("Inserted %d at index %d\n", key, index);
}

void display() {
    printf("\nHash Table (Linear Probing):\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (hashTable[i] == -1)
            printf("[%d] => EMPTY\n", i);
        else
            printf("[%d] => %d\n", i, hashTable[i]);
    }
}

int main() {
    int choice, key, n;

    printf("Enter table size: ");
    scanf("%d", &TABLE_SIZE);

    initHashTable();

    printf("\nMENU\n");
    printf("1) Insert\n");
    printf("2) Display\n");
    printf("3) Exit\n");

    while (1) {
        printf("\nEnter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Enter number of keys to insert: ");
                scanf("%d", &n);
                for (int i = 0; i < n; i++) {
                    printf("Enter key %d: ", i + 1);
                    scanf("%d", &key);
                    insert(key);
                }
                break;

            case 2:
                display();
                break;

            case 3:
                printf("Exiting...\n");
                free(hashTable);
                exit(0);

            default:
                printf("Invalid choice. Try again.\n");
        }
    }

    return 0;
}