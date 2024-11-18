BUILD_DIR := build
BIN_DIR := bin

.PHONY: all clean run

all:
	mkdir -p $(BUILD_DIR) 
	cd $(BUILD_DIR) && cmake .. && cmake --build .

clean:
	rm -rf -- $(BUILD_DIR) $(BIN_DIR)

run:
	@$(BUILD_DIR)/${BIN_DIR}/neuralc $(ARGS)
