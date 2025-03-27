using Alphalytics
using Documenter

DocMeta.setdocmeta!(Alphalytics, :DocTestSetup, :(using Alphalytics); recursive=true)

makedocs(;
    modules=[Alphalytics],
    authors="George Georgiev <georgegi86@gmail.com> and contributors",
    sitename="Alphalytics.jl",
    format=Documenter.HTML(;
        canonical="https://georgegee23.github.io/Alphalytics.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/georgegee23/Alphalytics.jl",
    devbranch="master",
)
