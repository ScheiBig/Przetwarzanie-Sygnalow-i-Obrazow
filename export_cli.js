const fs = require("fs")

const inputFilePath = process.argv.at(2)

if (inputFilePath == null || !inputFilePath.endsWith(".ipynb")) {
    throw new Error("You must pass input jupyter notebook path as first argument!")
}

const inputFile = fs.readFileSync(inputFilePath)

const outputFilePath = inputFilePath.replace(/\.ipynb$/, ".py")
if (fs.existsSync(outputFilePath) && process.argv.at(3) != "replace") {
    throw new Error(`Output file ${outputFilePath} already exists!
Pass "replace" as last argument to replace output file instead!`
    )
}

fs.writeFileSync(outputFilePath, "")

const input = JSON.parse(inputFile)

let prev_cell_type = undefined

for (const cell of input.cells) {
    if (cell.cell_type == "code") {
        if (prev_cell_type == "code") {
            fs.appendFileSync(outputFilePath, "\n\n#\n\n")
        } else if (prev_cell_type == "markdown") {
            fs.appendFileSync(outputFilePath, "\n")
        }

        for (const line of cell.source) {
            fs.appendFileSync(outputFilePath, line)
        }

        prev_cell_type = "code"
    } else if (cell.cell_type == "markdown") {

        if (prev_cell_type == "code") {
            fs.appendFileSync(outputFilePath, "\n\n\n")
        } else if (prev_cell_type = "markdown") {
            fs.appendFileSync(outputFilePath, "\n")
        }

        fs.appendFileSync(outputFilePath, "#".repeat(100) + "\n")
        fs.appendFileSync(outputFilePath, "#" + " ".repeat(98) + "#\n")

        for (const line of cell.source) {
            fs.appendFileSync(
                outputFilePath,
                "# " + line.replace(/\n$/, "").padEnd(96, " ") + " #\n"
            )
        }

        fs.appendFileSync(outputFilePath, "#" + " ".repeat(98) + "#\n")
        fs.appendFileSync(outputFilePath, "#".repeat(100) + "\n")

        prev_cell_type = "markdown"
    }
}

