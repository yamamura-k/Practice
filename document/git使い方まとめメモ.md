# 【MacOS/Linux向け】Gitについて

参考文献：エンジニアのためのGitの教科書(SHOEISHA,2016年), [gitのバージョンをbrewで管理](https://qiita.com/bakepotate/items/dc457a0046413ce1f965), [gitconfigの基本](https://qiita.com/shionit/items/fb4a1a30538f8d335b35)

## バージョン管理とは

これまでに加えられた変更の情報を記録し、後から古いバージョンを呼び出すことができるように

するためのもの。**集中型**(Subversionなど)と**分散型**(Gitなど)がある。

### できること

+ 履歴の保存
+ バージョンを戻す
+ バックアップ
+ 共同作業に便利

### 集中型(Centralized)

バージョン管理しているプロジェクトの全てが単一のサーバーに保存される。

+ 単一サーバーのため、メンバーの作業をある程度把握することができる。
+ サーバーの障害発生時に作業が停止してしまう。復旧しなかったら全てのデータが失われる。
+ 気軽にコミットできない(バグがあった場合に、全員にそのバグが反映されてしまうため)。

### 分散型(Distributed)

ローカルの二箇所に情報が保存される。

+ 障害発生時も作業が止まらない。
+ ローカルにアクセスすればいいだけなので、履歴の記録が高速に行われる。
+ ローカルブランチ作成可能。





## Gitの使い方



### Gitが管理するエリア

#### ワーキングディレクトリ

作業中のディレクトリ。変更履歴等が全てGitで監視される。

#### リポジトリ

ファイルをスナップショットとして保存するためにはコミットする必要がある。コミットによってファイルは実データとメタデータに分離される。リポジトリは**コミットされたデータを保存する**場所。実データとメタデータが格納される。(`git push`)

#### ステージングエリア

コミットしたいファイルだけを置いておく場所。施した変更ごとにファイルを一時的にまとめておくことができる。ここにあるファイルは同一のコミットメッセージをつけてコミットされる。(多分`git add` でここに入る。)



### 変更の監視

+ 新規作成(untracked)
+ Git監視下のファイルを監視下から外す(untracked)
+ 新しくファイルをGit監視下に追加(unmodified)
+ Git監視下のファイルを削除(modified)
+ Git監視下のファイルをステージング(staged)
+ ステージングされたファイルをコミット(unmodified)



### インストール

xcodeのインストール時にgitがインストールされるので注意。
macの場合は元からgitが入っているかもしれないが、バージョンが古い可能性があるので注意。

```bash
brew update
brew install git
```

(homebrewが既に入っていることを仮定しています)

### 更新

```bash
brew upgrade git
```

### 更新が反映されないとき

`.zshrc`か`.bash_profile`に以下を追記

```bash
export PATH=/usr/local/bin/git:$PATH
```

### 各種設定

gitの設定は、

+ 対象リポジトリ内の`.git/config` (`--local`オプションで指定される)
+ ホームディレクトリの`.gitconfig `(`--global`オプションで指定される)
+ Gitインストールディレクトリの`gitconfig `(`--system`オプションで指定される)

に書き込まれている。

コマンドラインからは、

```bash
git config --global 
```

で対象ファイルに設定を書き込むことができる。

+ ユーザー名`git config --global user.name "UserName"`
+ メールアドレス`git config --global user.email hogehoge@example.hoge`
+ 全てのコマンドに色つけ`git config --global color.ui true`
+ コマンドごとに指定するには、`color.ui`を`color.branch`、`color.diff`などのようにすれば良い。
+ エイリアス作成`git config --global alias.co checkout`など(coでcheckoutが実行される)

### 非追跡ファイルの設定

リポジトリのルートディレクトリかサブディレクトリに`.gitignore`を作成し、そこに無視したいファイルを書き込む。(正規表現使用可能)

+ リポジトリ内の特定の拡張子ファイル全部無視 = `*.拡張子`
+ リポジトリ内の特定の名前のフォルダ全部無視 = `フォルダ名/`

`.gitignore`に書き込む前に`git add`してしまった時は、`git rm --cached ファイル名`を実行し、`.gitignore`に追加すれば良い。

ホームディレクトリに`.gitignore_global`ファイルを作成すれば、システム全体で無視するファイルを設定できる。以下のコマンドを実行し、`.gitignore_global`の設定を有効化する：`git config --global core.excludesfile '~/.gitignore_global'`



## Gitコマンド集

#### リポジトリ作成(`git init` )

`.git`ファイルが作成され、カレントディレクトリがgit管理下に入る。`.git`ファイルを削除すればリポジトリじゃなくなる。

#### リポジトリの状態を確認(`git status`)

現在のブランチ、ステージに乗っているファイル、未追跡ファイルが表示される。

#### ステージングエリアにファイルを追加し、コミット対象にする(`git add`)

+ 全てのファイル追加`git add .`
+ 指定したファイル追加`git add <file1> <file2> <file3>`

#### ファイルの変更差分確認(`git diff`)

`--word-diff`オプションを追加すると単語単位での差分比較ができる。

#### 変更を保存(`git commit`)

このコマンドを実行することで、ステージングエリアにあるファイルを保存することができる。commit時にはコミットメッセージが必要。

+ コミットメッセージの追加は`-m`オプションを利用するか、`git commit`実行後に起動するエディタから入力する。
+ `-a`オプションで、変更があった全てのファイルをaddしてからcommitすることができる。(新規追加ファイルは含まれない)
+ `--amend`直前コミットの修正(追加忘れたファイルを追加してコミット)
+ `--no-edit`コミットメッセージ変更せずにコミット

**一つのコミットには一つの作業だけを含めること**

#### コミットの履歴を確認(`git log`)

デフォルトではGitの先端を表す[HEAD]が参照するコミットを表示する。そこから順に履歴を遡れる。

+ 表示するコミットの件数を制限`git log -n <limit>`
+ コミット内容を一行に`git log --oneline`
+ 追加行数、削除行数表示`git log --stat`
+ 完全な差分情報`git log -p`
+ 指定ファイルを含むコミットだけ表示`git log <file>`

#### コミット取り消し(`git reset <file>`)

+ ワーキングディレクトリの内容を直前のコミットまで戻す`git reset --hard`
+ 指定したコミットまで`git reset <commit id>`
+ 指定したコミットまでもどし、ワーキングディレクトリの状態も戻したい時`git reset --hard <commit id>`
+ アンステージングも`git reset <file>`で行える

#### Git管理下にないファイルを削除(`git clean`)

+ 削除対象の確認`git clean -n`
+ 強制削除`git clean -f`
+ ディレクトリ指定して確認`git clean -nd`

#### コミット、ファイルのチェックアウト(`git checkout <commit_id> <file>`)

+ ブランチ変更 `git checkout <branch>`
+ ファイルを過去の状態に戻す`git checkout <commit_id> <file>`
+ 過去のコミット状態に戻る`git checkout <commit_id>`

#### コミットを打ち消すコミット(`git revert <commit_id>`)

#### ファイル削除(`git rm`)

+ `--cached`オプションでステージングエリアから削除
+ `-n`オプションで削除対象ファイルを確認

#### ファイル移動・名前変更(`git mv`)

`git mv`はUnixコマンドの`mv`と`git add, git rm`を同時に行ってくれる。

#### ワーキングディレクトリから圧縮ファイル作成(`git archive`)

+ `-l`で圧縮方法確認

#### 作業内容を一時退避
+ `git stash`
+ `git stash list`
+ `git stash apply stash@{<number>}`

#### add取り消し
+ `git reset`
